//
//  ModelManager.swift
//  Aidoku
//
//  Created by Skitty on 6/25/25.
//

import CoreML

struct ModelList: Decodable {
    let models: [ModelInfo]
}

struct ModelInfo: Codable {
    var name: String?
    var info: String?
    var tags: [String]?
    var type: String?
    var miniOS: Int?
    var config: [String: JSONAnyValue]?
    var file: String
    var size: Int?
    var version: Int = 1

    init(file: String) {
        self.file = file
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        info = try container.decodeIfPresent(String.self, forKey: .info)
        tags = try container.decodeIfPresent([String].self, forKey: .tags)
        type = try container.decodeIfPresent(String.self, forKey: .type)
        miniOS = try container.decodeIfPresent(Int.self, forKey: .miniOS)
        config = try container.decodeIfPresent([String: JSONAnyValue].self, forKey: .config)
        file = try container.decode(String.self, forKey: .file)
        size = try container.decodeIfPresent(Int.self, forKey: .size)
        version = try container.decodeIfPresent(Int.self, forKey: .version) ?? 1
    }
}

actor ModelManager {
    static let shared = ModelManager()

    static private let modelListUrl = URL(string: "http://upscale.aidoku.app/models.json")!
    static private let supportedModelTypes: Set<String> = ["multiarray", "image"]

    private var imageModelCache: [String: ImageProcessingModel] = [:]
    private var cachedModelList: ModelList?
    private var didSeedBundledModels = false

    // download a model from the server and save to local disk
    func downloadModel(_ model: ModelInfo) async throws {
        let url = URL(string: model.file, relativeTo: Self.modelListUrl)!
        let fileName = (model.file as NSString).lastPathComponent
        let fileURL = try modelsDirectory().appendingPathComponent(fileName)

        var metadata = model
        metadata.file = fileName // ensure file field is just the local file name
        metadata.size = nil
        let metadataData = try JSONEncoder().encode(metadata)

        if fileName.hasSuffix(".mlpackage") {
            // download as a zip
            let tempZipURL = fileURL.appendingPathExtension("zip")
            let tempModelURL = fileURL.appendingPathExtension("download")
            let fm = FileManager.default
            defer {
                try? fm.removeItem(at: tempZipURL)
                try? fm.removeItem(at: tempModelURL)
            }
            let (data, _) = try await URLSession.shared.data(from: url.appendingPathExtension("zip"))
            try data.write(to: tempZipURL, options: .atomic)

            // unzip before removing an existing installation
            try fm.unzipItem(at: tempZipURL, to: tempModelURL)
            if fm.fileExists(atPath: fileURL.path) {
                try fm.removeItem(at: fileURL)
            }
            try fm.moveItem(at: tempModelURL, to: fileURL)
        } else {
            // download as a single file (.mlmodel)
            let (data, _) = try await URLSession.shared.data(from: url)
            try data.write(to: fileURL, options: .atomic)
        }

        imageModelCache[fileName] = nil

        let metadataURL = try metadataURL(forModelFile: fileName)
        try metadataData.write(to: metadataURL, options: .atomic)
    }

    // remove a downloaded model from disk
    func removeModel(withFile modelFile: String) {
        let fileName = (modelFile as NSString).lastPathComponent
        guard let fileURL = try? modelsDirectory().appendingPathComponent(fileName) else { return }

        // remove model file
        try? FileManager.default.removeItem(at: fileURL)
        // remove metadata file
        if let metadataURL = try? metadataURL(forModelFile: fileName) {
            try? FileManager.default.removeItem(at: metadataURL)
        }
        // remove from cache
        imageModelCache[fileName] = nil

        // don't bring a bundled model back on the next launch
        if BundledModels.all.contains(where: { $0.file == fileName }) {
            var removed = Set(UserDefaults.standard.stringArray(forKey: Self.removedBundledModelsKey) ?? [])
            removed.insert(fileName)
            UserDefaults.standard.set(Array(removed), forKey: Self.removedBundledModelsKey)
        }
    }

    // get installed models from disk
    // fetches extra info from the server if available
    func getInstalledModels() async -> [ModelInfo] {
        seedBundledModelsIfNeeded()

        guard
            let modelsDir = try? modelsDirectory(),
            let files = try? FileManager.default.contentsOfDirectory(atPath: modelsDir.path)
        else {
            return []
        }

        return files.compactMap { file in
            // only show models
            guard file.hasSuffix(".mlpackage") || file.hasSuffix(".mlmodel") || file.hasSuffix(".mlmodelc") else {
                return nil
            }

            let filePath = (modelsDir.path as NSString).appendingPathComponent(file)

            // load model metadata
            let metadataPath = filePath + ".json"
            var info: ModelInfo?
            if
                let data = try? Data(contentsOf: URL(fileURLWithPath: metadataPath)),
                let meta = try? JSONDecoder().decode(ModelInfo.self, from: data)
            {
                info = meta
                info?.file = file  // ensure file field is just the local file name
            }
            if info == nil {
                info = ModelInfo(file: file)
            }

            var fileSize: Int?
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: filePath, isDirectory: &isDir) {
                if isDir.boolValue {
                    // if it's a directory (.mlpackage), sum all contained file sizes
                    if let enumerator = FileManager.default.enumerator(atPath: filePath) {
                        var total: Int = 0
                        for case let subpath as String in enumerator {
                            let subfilePath = (filePath as NSString).appendingPathComponent(subpath)
                            if
                                let attrs = try? FileManager.default.attributesOfItem(atPath: subfilePath),
                                let subfileSize = attrs[.size] as? NSNumber
                            {
                                total += subfileSize.intValue
                            }
                        }
                        fileSize = total
                    }
                } else {
                    // if it's a file, just get its size
                    if
                        let attrs = try? FileManager.default.attributesOfItem(atPath: filePath),
                        let singleFileSize = attrs[.size] as? NSNumber
                    {
                        fileSize = singleFileSize.intValue
                    }
                }
            } else {
                fileSize = nil
            }
            info?.size = fileSize

            return info
        }
    }

    // get new models and updates from the server
    func getAvailableModels() async -> [ModelInfo]? {
        guard
            let modelList = try? await fetchModelList()
        else {
            return nil
        }

        let installedVersions = Dictionary(
            (await getInstalledModels()).map { ($0.file, $0.version) },
            uniquingKeysWith: { first, _ in first }
        )

        return modelList.models.filter {
            if let miniOS = $0.miniOS, miniOS > ProcessInfo.processInfo.operatingSystemVersion.majorVersion {
                // filter out models not supported on current iOS version
                return false
            }
            if let type = $0.type {
                let fileName = ($0.file as NSString).lastPathComponent
                return Self.supportedModelTypes.contains(type)
                    && $0.version > (installedVersions[fileName] ?? 0)
            } else {
                return false
            }
        }
    }

    // set the currently enabled model
    nonisolated func setEnabledModel(fileName: String?) {
        UserDefaults.standard.set(fileName, forKey: "Data.enabledModelFile")
    }

    // get currently enabled model file name
    nonisolated func getEnabledModelFileName() -> String? {
        UserDefaults.standard.string(forKey: "Data.enabledModelFile")
    }

    // get an instance of the currently enabled model
    func getEnabledModel() throws -> ImageProcessingModel? {
        guard let fileName = getEnabledModelFileName() else {
            return nil
        }
        seedBundledModelsIfNeeded()
        return try loadModel(info: .init(file: fileName))
    }
}

// MARK: - Bundled models
extension ModelManager {
    fileprivate static let removedBundledModelsKey = "Data.removedBundledModels"

    // copy models shipped inside the app into the models directory (once per launch)
    // so they show up in the model list like downloaded ones
    fileprivate func seedBundledModelsIfNeeded() {
        guard !didSeedBundledModels else { return }
        didSeedBundledModels = true

        guard let modelsDir = try? modelsDirectory() else { return }
        let fm = FileManager.default
        let removed = Set(UserDefaults.standard.stringArray(forKey: Self.removedBundledModelsKey) ?? [])

        for info in BundledModels.all where !removed.contains(info.file) {
            let base = (info.file as NSString).deletingPathExtension
            guard let bundled = Bundle.main.url(forResource: base, withExtension: "mlmodelc") else { continue }

            let destination = modelsDir.appendingPathComponent(info.file)
            let metadataURL = modelsDir.appendingPathComponent(info.file + ".json")

            // skip if the installed copy is already up to date
            if
                fm.fileExists(atPath: destination.path),
                let data = try? Data(contentsOf: metadataURL),
                let installed = try? JSONDecoder().decode(ModelInfo.self, from: data),
                installed.version >= info.version
            {
                continue
            }

            let temporary = destination.appendingPathExtension("seeding")
            do {
                try? fm.removeItem(at: temporary)
                try fm.copyItem(at: bundled, to: temporary)
                if fm.fileExists(atPath: destination.path) {
                    try fm.removeItem(at: destination)
                }
                try fm.moveItem(at: temporary, to: destination)
                try JSONEncoder().encode(info).write(to: metadataURL, options: .atomic)
                imageModelCache[info.file] = nil
            } catch {
                try? fm.removeItem(at: temporary)
                LogManager.logger.error("Failed to install bundled model \(info.file): \(error)")
            }
        }
    }
}

// MARK: - Helpers
extension ModelManager {
    // get the directory where models are stored
    private func modelsDirectory() throws -> URL {
        let modelsDir = FileManager.default.documentDirectory.appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: modelsDir.path) {
            try FileManager.default.createDirectory(
                at: modelsDir,
                withIntermediateDirectories: true
            )
        }
        return modelsDir
    }

    // get the list of models from the server
    private func fetchModelList() async throws -> ModelList {
        if let cachedModelList {
            return cachedModelList
        }
        let modelList: ModelList = try await URLSession.shared.object(from: Self.modelListUrl)
        cachedModelList = modelList
        return modelList
    }

    // get the url to an installed model's metadata file
    private func metadataURL(forModelFile fileName: String) throws -> URL {
        try modelsDirectory().appendingPathComponent(fileName + ".json")
    }

    // load a usable instance of an image model from the disk
    private func loadModel(info: ModelInfo) throws -> ImageProcessingModel? {
        let modelsDir = try modelsDirectory()
        let fileName = (info.file as NSString).lastPathComponent
        let fileURL = modelsDir.appendingPathComponent(fileName)

        // return cached model if available
        if let cached = imageModelCache[fileName] {
            return cached
        }

        // get model type from given info, or try loading from metadata
        var info = info
        if info.type == nil {
            let metadataURL = try metadataURL(forModelFile: fileName)
            if
                let data = try? Data(contentsOf: metadataURL),
                let meta = try? JSONDecoder().decode(ModelInfo.self, from: data)
            {
                info = meta
            }
        }

        guard let modelType = info.type else { return nil }

        // load coreml model
        // bundled models are already compiled
        let compiledUrl = fileURL.pathExtension == "mlmodelc" ? fileURL : try MLModel.compileModel(at: fileURL)
        let mlModel = try MLModel(contentsOf: compiledUrl)
        let model: ImageProcessingModel?

        let config = info.config?.compactMapValues { $0.toRaw() } ?? [:]

        switch modelType.lowercased() {
            case "multiarray":
                model = MultiArrayModel(model: mlModel, config: config)
            case "image":
                model = ImageModel(model: mlModel, config: config)
            default:
                model = nil
        }
        if let model {
            imageModelCache[fileName] = model
        }
        return model
    }
}
