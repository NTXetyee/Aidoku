//
//  DownloadBackgroundExecution.swift
//  Aidoku
//
//  Keeps track of whether the app can currently do download work, and keeps it alive while it does.
//
//  When the app leaves the foreground (switching apps, locking the screen) iOS suspends it within
//  seconds unless it holds a background assertion or a continued processing task. Requests that are
//  in flight at that moment fail with network errors once the app wakes again, which used to burn
//  through the page retries and mark whole chapters as failed. This type lets the download code
//  hold a short background assertion, tell whether it is about to be suspended, and wait for the
//  app to come back instead of giving up.
//

import Foundation
import UIKit

final class DownloadBackgroundExecution: @unchecked Sendable {
    static let shared = DownloadBackgroundExecution()

    private let lock = NSLock()

    private var foreground = true
    private var continuedTaskActive = false
    private var assertionActive = false
    private var foregroundWaiters: [UUID: CheckedContinuation<Void, Never>] = [:]

    // main actor only
    private var assertionId: UIBackgroundTaskIdentifier = .invalid
    private var started = false

    /// Called when the app comes back to the foreground.
    private var onForeground: (@Sendable () -> Void)?
    /// Called when the background assertion runs out and the app is about to be suspended.
    private var onSuspend: (@Sendable () -> Void)?

    private init() {}

    /// Whether the app is in the foreground.
    var isForeground: Bool {
        locked { foreground }
    }

    /// Whether a continued processing task is keeping the app running in the background.
    var isContinuedTaskActive: Bool {
        locked { continuedTaskActive }
    }

    /// Whether download work can keep going for more than a few seconds.
    var canRunIndefinitely: Bool {
        locked { foreground || continuedTaskActive }
    }

    /// Whether download work can happen right now, even if only for a few more seconds.
    var canRunNow: Bool {
        locked { foreground || continuedTaskActive || assertionActive }
    }

    func setContinuedTaskActive(_ active: Bool) {
        locked { continuedTaskActive = active }
    }

    func setHandlers(
        onForeground: (@Sendable () -> Void)?,
        onSuspend: (@Sendable () -> Void)?
    ) {
        locked {
            self.onForeground = onForeground
            self.onSuspend = onSuspend
        }
    }

    /// Start observing app lifecycle notifications.
    @MainActor
    func start() {
        guard !started else { return }
        started = true

        let isForeground = UIApplication.shared.applicationState != .background
        locked { foreground = isForeground }

        let center = NotificationCenter.default
        center.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.setForeground(false)
        }
        center.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.setForeground(true)
        }
        center.addObserver(
            forName: UIApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.setForeground(true)
        }
    }

    private func setForeground(_ value: Bool) {
        let (waiters, handler, changed): ([CheckedContinuation<Void, Never>], (@Sendable () -> Void)?, Bool) = locked {
            let changed = foreground != value
            foreground = value
            guard value else { return ([], nil, changed) }
            let waiters = Array(foregroundWaiters.values)
            foregroundWaiters.removeAll()
            return (waiters, onForeground, changed)
        }
        for waiter in waiters {
            waiter.resume()
        }
        if value && changed {
            handler?()
        }
    }

    /// Suspends until the app is in the foreground, or returns right away if it already is.
    func waitForForeground() async {
        let id = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                let resumeNow = locked {
                    if foreground || Task.isCancelled {
                        return true
                    }
                    foregroundWaiters[id] = continuation
                    return false
                }
                if resumeNow {
                    continuation.resume()
                }
            }
        } onCancel: {
            let waiter = locked { foregroundWaiters.removeValue(forKey: id) }
            waiter?.resume()
        }
    }

    /// Waits for the foreground when the app has no way of running in the background right now.
    func waitUntilRunnable() async {
        guard !canRunNow else { return }
        await waitForForeground()
    }
}

// MARK: - Background assertion
extension DownloadBackgroundExecution {
    /// Ask iOS for extra time to keep downloading after the app leaves the foreground.
    @MainActor
    func beginAssertion() {
        guard assertionId == .invalid else { return }
        locked { assertionActive = true }
        assertionId = UIApplication.shared.beginBackgroundTask(withName: "Aidoku.download") { [weak self] in
            guard let self else { return }
            // out of time, let downloads stop cleanly before the app gets suspended
            let handler = self.locked {
                self.assertionActive = false
                return self.continuedTaskActive ? nil : self.onSuspend
            }
            handler?()
            self.endAssertion()
        }
    }

    @MainActor
    func endAssertion() {
        locked { assertionActive = false }
        guard assertionId != .invalid else { return }
        UIApplication.shared.endBackgroundTask(assertionId)
        assertionId = .invalid
    }
}

private extension DownloadBackgroundExecution {
    func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}
