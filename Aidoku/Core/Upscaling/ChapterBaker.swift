//
//  ChapterBaker.swift
//  Aidoku
//
//  Upscales the pages of a downloaded chapter once, so the reader doesn't have to do it live.
//

import CoreGraphics
import Foundation
import ImageIO
import ZIPFoundation

struct BakeMarker: Codable {
    var model: String?
    var maxHeight: Int
    var format: String
    var date: Date
    var bakedPages: [String]
}

actor ChapterBaker {
    static let shared = ChapterBaker()

    static let markerName = ".baked.json"

    private static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "gif", "heic", "avif"]

    // only one chapter is baked at a time, even if multiple sources are downloading
    private var isBusy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Upscales every eligible image in the directory, replacing it with a HEIC file.
    /// Returns the number of pages that were baked.
    @discardableResult
    func bake(
        directory: URL,
        progress: @escaping @Sendable (_ done: Int, _ total: Int) -> Void
    ) async -> Int {
        await acquire()
        defer { release() }

        guard let model = try? await ModelManager.shared.getEnabledModel() else {
            return 0
        }
        let modelName = ModelManager.shared.getEnabledModelFileName()
        let maxHeight = UserDefaults.standard.integer(forKey: "Reader.upscaleMaxHeight")

        let files = ((try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? [])
            .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }

        guard !files.isEmpty else { return 0 }

        var baked: [String] = []
        progress(0, files.count)

        for (index, file) in files.enumerated() {
            if Task.isCancelled { break }
            await waitForCoolDown()

            if let name = await bakePage(file: file, model: model, maxHeight: maxHeight) {
                baked.append(name)
            }
            progress(index + 1, files.count)
        }

        guard !baked.isEmpty else { return 0 }

        let marker = BakeMarker(
            model: modelName,
            maxHeight: maxHeight,
            format: "heic",
            date: Date(),
            bakedPages: baked
        )
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(marker).write(to: directory.appendingPathComponent(Self.markerName))
        } catch {
            LogManager.logger.error("Failed to write bake marker: \(error)")
        }
        return baked.count
    }

    /// Names of the pages in a downloaded chapter (folder or .cbz) that were already upscaled.
    nonisolated static func bakedPages(forChapterAt url: URL) -> Set<String> {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        if url.pathExtension.lowercased() == "cbz" {
            guard
                let archive = try? Archive(url: url, accessMode: .read),
                let entry = archive[markerName]
            else { return [] }
            var data = Data()
            _ = try? archive.extract(entry, consumer: { data.append($0) })
            guard let marker = try? decoder.decode(BakeMarker.self, from: data) else { return [] }
            return Set(marker.bakedPages)
        }

        let markerUrl = url.appendingPathComponent(markerName)
        guard
            let data = try? Data(contentsOf: markerUrl),
            let marker = try? decoder.decode(BakeMarker.self, from: data)
        else { return [] }
        return Set(marker.bakedPages)
    }
}

// MARK: Page processing
extension ChapterBaker {
    private func bakePage(file: URL, model: ImageProcessingModel, maxHeight: Int) async -> String? {
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil) else { return nil }
        // check size without decoding the whole image
        if
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let height = properties[kCGImagePropertyPixelHeight] as? Int,
            height >= maxHeight
        {
            return nil
        }
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        guard cgImage.height < maxHeight else { return nil }

        guard let output = await model.process(cgImage) else {
            LogManager.logger.error("Upscaling model failed to process \(file.lastPathComponent)")
            return nil
        }

        let outputUrl = file.deletingPathExtension().appendingPathExtension("heic")
        let tempUrl = file.deletingLastPathComponent().appendingPathComponent(".baking_\(outputUrl.lastPathComponent)")

        guard
            let destination = CGImageDestinationCreateWithURL(tempUrl as CFURL, "public.heic" as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(
            destination,
            output,
            [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: tempUrl)
            LogManager.logger.error("Failed to encode HEIC for \(file.lastPathComponent)")
            return nil
        }

        do {
            if outputUrl != file {
                try FileManager.default.removeItem(at: file)
            }
            if FileManager.default.fileExists(atPath: outputUrl.path) {
                try FileManager.default.removeItem(at: outputUrl)
            }
            try FileManager.default.moveItem(at: tempUrl, to: outputUrl)
        } catch {
            try? FileManager.default.removeItem(at: tempUrl)
            LogManager.logger.error("Failed to save baked page \(outputUrl.lastPathComponent): \(error)")
            return nil
        }
        return outputUrl.lastPathComponent
    }

    // pause while the device is hot
    private func waitForCoolDown() async {
        while !Task.isCancelled {
            let state = ProcessInfo.processInfo.thermalState
            guard state == .serious || state == .critical else { return }
            try? await Task.sleep(nanoseconds: 5_000_000_000)
        }
    }

    private func acquire() async {
        if isBusy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            isBusy = true
        }
    }

    private func release() {
        if waiters.isEmpty {
            isBusy = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}
