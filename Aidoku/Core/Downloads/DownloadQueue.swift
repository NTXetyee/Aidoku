//
//  DownloadQueue.swift
//  Aidoku
//
//  Created by Skitty on 5/13/22.
//

import AidokuRunner
@preconcurrency import BackgroundTasks
import Foundation

// stores queued and active downloads
// creates a downloadtask for every source
// only one chapter per source is downloaded at a time
actor DownloadQueue {
    private let cache: DownloadCache
    private var onCompletion: (() -> Void)?

    private(set) var queue: [String: [Download]] = [:] // all queued downloads stored under source id
    private var tasks: [String: DownloadTask] = [:] // tasks for each source
    private var progressBlocks: [ChapterIdentifier: (Int, Int) -> Void] = [:]

    private var paused = false
    /// Paused because iOS took away background time, rather than by the user.
    private var pausedBySystem = false
    private var registeredTask = false
    private var totalDownloads: Int = 0
    private var completedDownloads: Int = 0
    /// Progress of the chapter currently downloading, from 0 to 1.
    private var currentChapterFraction: Double = 0
    private var bgTask: ProgressReporting?
    /// A continued processing task was submitted and its launch handler hasn't run yet.
    private var bgTaskPending = false
    /// Resumed when the background task should end (queue empty, paused, or expired).
    private var bgTaskWaiter: CheckedContinuation<Void, Never>?
    private var sendCancelNotification = true

    /// Units of background task progress per chapter, so page progress can be reported too.
    private static let unitsPerChapter: Int64 = 100

    private static let taskIdentifier = (Bundle.main.bundleIdentifier ?? "") + ".download"

    init(cache: DownloadCache, onCompletion: (() -> Void)? = nil) {
        self.cache = cache
        self.onCompletion = onCompletion

        DownloadBackgroundExecution.shared.setHandlers(
            onForeground: { [weak self] in
                Task { await self?.appDidEnterForeground() }
            },
            onSuspend: { [weak self] in
                Task { await self?.appWillBeSuspended() }
            }
        )
        Task { @MainActor in
            DownloadBackgroundExecution.shared.start()
        }
    }

    func setOnCompletion(_ onCompletion: (() -> Void)?) {
        self.onCompletion = onCompletion
    }

    func start() async {
        paused = false
        pausedBySystem = false

        guard !queue.isEmpty else { return }

        // a continued processing task is running or about to, it will pick up the queue
        if bgTask != nil || bgTaskPending {
            await initAndResumeTasks()
            return
        }

#if !targetEnvironment(simulator)
        if
            #available(iOS 26.0, *),
            AppSettings.downloads.background.get(),
            !ProcessInfo.processInfo.isMacCatalystApp,
            // continued processing tasks can only be started from the foreground
            DownloadBackgroundExecution.shared.isForeground
        {
            await register()

            let request = BGContinuedProcessingTaskRequest(
                identifier: Self.taskIdentifier,
                title: NSLocalizedString("DOWNLOADING"),
                subtitle: NSLocalizedString("PROCESSING_QUEUE")
            )
            // run now or not at all, otherwise downloads could sit in the system queue doing nothing
            request.strategy = .fail
            bgTaskPending = true
            do {
                try await BGTaskScheduler.shared.submit(request: request)
            } catch {
                bgTaskPending = false
                LogManager.logger.error("Failed to start background downloading: \(error)")
            }
        }
#endif

        // start right away either way; the background task only keeps the app alive
        await initAndResumeTasks()
    }

    private func initAndResumeTasks() async {
        guard !queue.isEmpty else { return }
        // keep running for a bit if the app leaves the foreground
        await DownloadBackgroundExecution.shared.beginAssertion()
        for (sourceKey, downloads) in queue {
            if tasks[sourceKey] == nil {
                let task = DownloadTask(id: sourceKey, cache: cache, downloads: downloads)
                await task.setDelegate(delegate: self)
                tasks[sourceKey] = task
            }
            await tasks[sourceKey]?.resume()
        }
    }

    func resume() async {
        paused = false
        pausedBySystem = false

        await start()

        await withTaskGroup(of: Void.self) { group in
            for task in tasks.values {
                group.addTask { await task.resume() }
            }
        }
    }

    func pause(bySystem: Bool = false) async {
        paused = true
        pausedBySystem = bySystem

        await withTaskGroup(of: Void.self) { group in
            for task in tasks.values {
                group.addTask { await task.pause() }
            }
        }
        saveQueueState()

        // nothing left to do in the background, let the system task go
        detachBackgroundTask()
        await DownloadBackgroundExecution.shared.endAssertion()
    }

    /// The app came back to the foreground.
    private func appDidEnterForeground() async {
        guard !queue.isEmpty else { return }
        if pausedBySystem {
            // downloads were stopped because the app was about to be suspended, pick them back up
            LogManager.logger.info("Resuming downloads paused while in the background")
            await DownloadManager.shared.resumeDownloads()
        } else if !paused {
            // renew the background assertion that may have run out
            await initAndResumeTasks()
        }
    }

    /// The background assertion ran out and the app is about to be suspended.
    private func appWillBeSuspended() async {
        guard !queue.isEmpty, !paused, bgTask == nil else { return }
        LogManager.logger.info("Out of background time, pausing downloads until the app is opened")
        await pause(bySystem: true)
        NotificationCenter.default.post(name: .downloadsPaused, object: nil)
    }

    @discardableResult
    func add(
        chapters: [AidokuRunner.Chapter],
        manga: AidokuRunner.Manga,
        autoStart: Bool = true,
        upscale: Bool = false
    ) async -> [Download] {
        var downloads: [Download] = []
        for chapter in chapters {
            let identifier = ChapterIdentifier(
                sourceKey: manga.sourceKey,
                mangaKey: manga.key,
                chapterKey: chapter.key
            )
            guard !(await cache.isChapterDownloaded(identifier: identifier)) else {
                continue
            }

            // create tmp directory so we know it's queued
            let tmpDirectory = cache.tmpDirectory(for: identifier)
            tmpDirectory.removeItem() // remove in case it exists from a previous failed download
            tmpDirectory.createDirectory()

            let download = Download.from(manga: manga, chapter: chapter, upscale: upscale)
            downloads.append(download)
            if queue[manga.sourceKey] == nil {
                queue[manga.sourceKey] = [download]
            } else {
                queue[manga.sourceKey]?.append(download)
                await tasks[manga.sourceKey]?.add(download: download)
            }
        }
        totalDownloads += downloads.count
        bgTask?.progress.totalUnitCount = Int64(totalDownloads) * Self.unitsPerChapter
        if autoStart {
            await start()
        }
        saveQueueState()
        return downloads
    }

    func cancelDownload(for chapter: ChapterIdentifier) async {
        if let task = tasks[chapter.sourceKey] {
            await task.cancel(chapter: chapter)
        } else {
            // no longer in queue but the tmp download directory still exists, so we should remove it
            cache.tmpDirectory(for: chapter).removeItem()
        }
        saveQueueState()
    }

    func cancelDownloads(for chapters: [ChapterIdentifier]) async {
        // disable individual download cancelled notifications
        sendCancelNotification = false
        defer { sendCancelNotification = true }
        for chapter in chapters {
            if let task = tasks[chapter.sourceKey] {
                await task.cancel(chapter: chapter)
            } else {
                cache.tmpDirectory(for: chapter).removeItem()
            }
            if let queueItem = queue[chapter.sourceKey]?.firstIndex(where: {
                $0.chapterIdentifier == chapter
            }) {
                queue[chapter.sourceKey]?.remove(at: queueItem)
            }
        }
        NotificationCenter.default.post(name: .downloadsCancelled, object: chapters)
        saveQueueState()
    }

    func cancelDownloads(for manga: MangaIdentifier) async {
        if let task = tasks[manga.sourceKey] {
            await task.cancel(manga: manga)
        } else {
            cache.directory(for: manga)
                .contentsIncludingHidden
                .filter {
                    $0.lastPathComponent.hasPrefix(DownloadCache.tmpDirectoryPrefix)
                        && !cache.hasFailureMarker(inTmpDirectory: $0)
                }
                .forEach { $0.removeItem() }
        }
        saveQueueState()
    }

    func cancelAll() async {
        sendCancelNotification = false
        defer { sendCancelNotification = true }
        for task in tasks {
            await task.value.cancel()
        }
        queue = [:]
        NotificationCenter.default.post(name: .downloadsCancelled, object: nil)
        saveQueueState()
        await queueEmptied()
    }

    // register callback for download progress change
    func onProgress(for chapter: ChapterIdentifier, block: @escaping (Int, Int) -> Void) {
        progressBlocks[chapter] = block
    }

    func removeProgressBlock(for chapter: ChapterIdentifier) {
        progressBlocks.removeValue(forKey: chapter)
    }

    func saveQueueState() {
        let queueData = try? JSONEncoder().encode(queue)
        UserDefaults.standard.set(queueData, forKey: "Data.downloadQueueState")
    }

    func loadQueueState() async {
        guard
            let queueData = UserDefaults.standard.data(forKey: "Data.downloadQueueState"),
            let queueState = try? JSONDecoder().decode([String: [Download]].self, from: queueData)
        else {
            return
        }
        queue = queueState
        if !queue.isEmpty {
            await start()
        }
    }

    func hasQueuedDownloads() -> Bool {
        !queue.isEmpty
    }

    func isRunning() async -> Bool {
        for task in tasks where await task.value.running {
            return true
        }
        return false
    }
}

extension DownloadQueue {
    private func setBackgroundTask(_ task: ProgressReporting?) {
        bgTask = task
        bgTaskPending = false
        DownloadBackgroundExecution.shared.setContinuedTaskActive(task != nil)
        totalDownloads = queue.values.reduce(0) { $0 + $1.count }
        completedDownloads = 0
        currentChapterFraction = 0
        bgTask?.progress.totalUnitCount = Int64(max(totalDownloads, 1)) * Self.unitsPerChapter
        bgTask?.progress.completedUnitCount = 0
    }

    /// Stops using the current background task and lets its launch handler finish.
    private func detachBackgroundTask() {
        if bgTask != nil {
            bgTask = nil
            DownloadBackgroundExecution.shared.setContinuedTaskActive(false)
        }
        endBackgroundTask()
    }

    /// Clears the background task if it's still the given one.
    private func clearBackgroundTask(_ task: BGTask) {
        guard let current = bgTask as? BGTask, current === task else { return }
        setBackgroundTask(nil)
    }

    /// Lets the background task's launch handler finish.
    private func endBackgroundTask() {
        bgTaskWaiter?.resume()
        bgTaskWaiter = nil
    }

    private func waitForBackgroundTaskEnd() async {
        guard !queue.isEmpty, !paused else { return }
        await withCheckedContinuation { continuation in
            bgTaskWaiter?.resume()
            bgTaskWaiter = continuation
        }
    }

    /// Called whenever the queue may have run out of downloads.
    private func queueEmptied() async {
        guard queue.isEmpty else { return }
        endBackgroundTask()
        await DownloadBackgroundExecution.shared.endAssertion()
    }

    private func updateBackgroundProgress() {
        guard let bgTask else { return }
        let total = Int64(max(totalDownloads, 1)) * Self.unitsPerChapter
        let completed = Int64(completedDownloads) * Self.unitsPerChapter
            + Int64(currentChapterFraction * Double(Self.unitsPerChapter - 1))
        if bgTask.progress.totalUnitCount != total {
            bgTask.progress.totalUnitCount = total
        }
        // never move backwards, the system watches this to decide whether the task is stuck
        let value = min(total, completed)
        if value > bgTask.progress.completedUnitCount {
            bgTask.progress.completedUnitCount = value
        }
    }

#if !targetEnvironment(simulator)
    @available(iOS 26.0, *)
    private func register() async {
        guard !registeredTask else { return }
        registeredTask = true

        BGTaskScheduler.shared.register(forTaskWithIdentifier: Self.taskIdentifier, using: nil) { @Sendable [weak self] task in
            guard let task = task as? BGContinuedProcessingTask else { return }
            guard let self else {
                task.setTaskCompleted(success: false)
                return
            }

            let completion = BackgroundTaskCompletion(task: task)

            task.expirationHandler = {
                // the system or the user stopped the task; pause and pick up again when the app opens
                Task {
                    await self.backgroundTaskExpired(task)
                    completion.complete(success: false)
                }
            }

            Task { @Sendable in
                await self.setBackgroundTask(task)
                await self.initAndResumeTasks()

                // wait until downloads complete, are paused, or the task expires
                await self.waitForBackgroundTaskEnd()

                let success = await self.queue.isEmpty
                await self.clearBackgroundTask(task)
                completion.complete(success: success)
            }
        }
    }

    private func backgroundTaskExpired(_ task: BGTask) async {
        LogManager.logger.info("Background download task expired")
        guard let current = bgTask as? BGTask, current === task else { return }
        let wasRunning = !paused
        setBackgroundTask(nil)
        endBackgroundTask()
        if wasRunning && !queue.isEmpty {
            if DownloadBackgroundExecution.shared.isForeground {
                // still in the app, keep going without the system task
                await initAndResumeTasks()
            } else {
                await pause(bySystem: true)
                NotificationCenter.default.post(name: .downloadsPaused, object: nil)
            }
        }
    }
#endif
}

/// Makes sure a background task is only marked completed once.
private final class BackgroundTaskCompletion: @unchecked Sendable {
    private let task: BGTask
    private let lock = NSLock()
    private var completed = false

    init(task: BGTask) {
        self.task = task
    }

    func complete(success: Bool) {
        lock.lock()
        let shouldComplete = !completed
        completed = true
        lock.unlock()
        if shouldComplete {
            task.setTaskCompleted(success: success)
        }
    }
}

// MARK: - Task Delegate
extension DownloadQueue: DownloadTaskDelegate {
    func taskCancelled(task: DownloadTask) async {
        await taskFinished(task: task)
    }

    func taskPaused(task _: DownloadTask) async {}

    func taskFinished(task: DownloadTask) async {
        tasks.removeValue(forKey: task.id)
        queue.removeValue(forKey: task.id)
        saveQueueState()
        await queueEmptied()
    }

    func downloadFinished(download: Download) async {
        await downloadCancelled(download: download)
        progressBlocks.removeValue(forKey: download.chapterIdentifier)
        onCompletion?()
        NotificationCenter.default.post(name: .downloadFinished, object: download)
    }

    func downloadFailed(download: Download) async {
        await downloadCancelled(download: download)
        progressBlocks.removeValue(forKey: download.chapterIdentifier)
        onCompletion?()
        NotificationCenter.default.post(name: .downloadFailed, object: download)
    }

    func downloadCancelled(download: Download) async {
        var sourceDownloads = queue[download.chapterIdentifier.sourceKey] ?? []
        sourceDownloads.removeAll { $0 == download }
        if sourceDownloads.isEmpty {
            queue.removeValue(forKey: download.chapterIdentifier.sourceKey)
        } else {
            queue[download.chapterIdentifier.sourceKey] = sourceDownloads
        }
        saveQueueState()
        progressBlocks.removeValue(forKey: download.chapterIdentifier)
        if sendCancelNotification {
            NotificationCenter.default.post(name: .downloadCancelled, object: download)
        }

        completedDownloads += 1
        currentChapterFraction = 0
        updateBackgroundProgress()

        if #available(iOS 26.0, *) {
            if !paused, let task = bgTask as? BGContinuedProcessingTask {
                task.updateTitle(
                    NSLocalizedString("DOWNLOADING"),
                    subtitle: String(format: NSLocalizedString("%i_OF_%i"), min(completedDownloads, totalDownloads), totalDownloads)
                )
            }
        }

        await queueEmptied()
    }

    func downloadProgressChanged(download: Download) async {
        if let index = queue[download.chapterIdentifier.sourceKey]?.firstIndex(where: { $0 == download }) {
            queue[download.chapterIdentifier.sourceKey]?[index] = download
        }
        if let block = progressBlocks[download.chapterIdentifier] {
            block(download.progress, download.total)
        }
        NotificationCenter.default.post(name: .downloadProgressed, object: download)

        // report page progress to the background task so the system knows it isn't stuck
        if bgTask != nil, download.total > 0 {
            let pageFraction = min(1, Double(download.progress) / Double(download.total))
            let fraction: Double = if download.upscale == true {
                // first half is downloading pages, second half is upscaling them
                download.baking == true ? 0.5 + pageFraction / 2 : pageFraction / 2
            } else {
                pageFraction
            }
            currentChapterFraction = max(currentChapterFraction, fraction)
            updateBackgroundProgress()
        }
    }
}
