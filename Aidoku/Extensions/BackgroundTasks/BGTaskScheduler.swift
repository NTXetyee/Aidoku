//
//  BGTaskScheduler.swift
//  Aidoku
//
//  Created by skitty on 8/16/26.
//

import BackgroundTasks

extension BGTaskScheduler {
    func submit(request: BGTaskRequest) async throws {
// submitTaskRequest only exists in the iOS 27 SDK (Xcode 27 / Swift 6.4); older SDKs use the sync call
#if compiler(>=6.4)
        if #available(iOS 27.0, *) {
            try await submitTaskRequest(request)
        } else {
            try submit(request)
        }
#else
        try submit(request)
#endif
    }
}
