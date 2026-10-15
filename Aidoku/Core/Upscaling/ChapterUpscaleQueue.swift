//
//  ChapterUpscaleQueue.swift
//  Aidoku
//
//  Upscales chapters that are already downloaded, one at a time.
//

import Foundation
import ZIPFoundation

struct ChapterUpscaleProgress {
    let chapter: ChapterIdentifier
    let done: Int
    let total: Int
}

actor ChapterUpscaleQueue {
    static let shared = ChapterUpscaleQueue()

    // prefix keeps working files hidden from the download cache and treated like temporary downloads
    private static let workPrefix = ".tmp_upscale_"

    private var pending: [ChapterIdentifier] = []
    private var current: ChapterIdentifier?
    private var currentTask: Task<Void, Never>?

    func enqueue(_ chapters: [ChapterIdentifier]) {
        for chapter in chapters where chapter != current && !pending.contains(chapter) {
            pending.append(chapter)
        }
        runNextIfIdle()
    }

    func cancel(_ chapter: ChapterIdentifier) {
        pending.removeAll { $0 == chapter }
        if current == chapter {
            currentTask?.cancel()
        }
    }

    func isQueued(_ chapter: ChapterIdentifier) -> Bool {
        current == chapter || pending.contains(chapter)
    }

    private func runNextIfIdle() {
        guard current == nil, !pending.isEmpty else { return }
        let chapter = pending.removeFirst()
        current = chapter
        currentTask = Task {
            await self.process(chapter)
            self.finishCurrent()
        }
    }

    private func finishCurrent() {
        current = nil
        currentTask = nil
        runNextIfIdle()
    }
}

extension ChapterUpscaleQueue {
    private func process(_ chapter: ChapterIdentifier) async {
        defer {
            NotificationCenter.default.post(name: .chapterUpscaleFinished, object: chapter)
        }

        let fm = FileManager.default
        let chapterDirectory = DownloadManager.directory
            .appendingSafePathComponent(chapter.sourceKey)
            .appendingSafePathComponent(chapter.mangaKey)
            .appendingSafePathComponent(chapter.chapterKey)
        let archive = chapterDirectory.appendingPathExtension("cbz")
        let isArchive = archive.exists
        guard isArchive || chapterDirectory.exists else { return }

        // already upscaled
        let existing = isArchive ? archive : chapterDirectory
        guard ChapterBaker.bakedPages(forChapterAt: existing).isEmpty else { return }

        let mangaDirectory = chapterDirectory.deletingLastPathComponent()
        let name = chapterDirectory.lastPathComponent
        let work = mangaDirectory.appendingPathComponent(Self.workPrefix + name)
        let tmpArchive = mangaDirectory.appendingPathComponent(Self.workPrefix + name + ".cbz")
        work.removeItem()
        tmpArchive.removeItem()
        defer {
            work.removeItem()
            tmpArchive.removeItem()
        }

        do {
            if isArchive {
                try fm.unzipItem(at: archive, to: work)
            } else {
                try fm.copyItem(at: chapterDirectory, to: work)
            }
        } catch {
            LogManager.logger.error("Failed to prepare chapter for upscaling: \(error)")
            return
        }

        let baked = await ChapterBaker.shared.bake(directory: work) { done, total in
            NotificationCenter.default.post(
                name: .chapterUpscaleProgressed,
                object: ChapterUpscaleProgress(chapter: chapter, done: done, total: total)
            )
        }
        guard baked > 0, !Task.isCancelled else { return }

        do {
            if isArchive {
                try fm.zipItem(at: work, to: tmpArchive, shouldKeepParent: false)
                // make sure the marker made it into the archive
                if
                    ChapterBaker.bakedPages(forChapterAt: tmpArchive).isEmpty,
                    let zip = try? Archive(url: tmpArchive, accessMode: .update)
                {
                    try? zip.addEntry(with: ChapterBaker.markerName, relativeTo: work)
                }
                _ = try fm.replaceItemAt(archive, withItemAt: tmpArchive)
            } else {
                _ = try fm.replaceItemAt(chapterDirectory, withItemAt: work)
            }
        } catch {
            LogManager.logger.error("Failed to save upscaled chapter: \(error)")
        }
    }
}
