//
//  BundledModels.swift
//  Aidoku
//
//  Upscaling models that ship inside the app (see tools/upscale-models).
//

import Foundation

enum BundledModels {
    // bump `version` when a model file changes so installed copies are replaced
    private static let json = """
    [
      {
        "name": "AnimeVideo-v3 (2x, fast)",
        "info": "Compact Real-ESRGAN model for anime and manga line art. Much faster than Real-ESRGAN. Real-ESRGAN by Xintao Wang et al. (BSD-3-Clause).",
        "tags": ["2x", "fast", "anime"],
        "type": "multiarray",
        "file": "AnimeVideoV3_x2.mlmodelc",
        "version": 1,
        "config": {
          "inputName": "input",
          "outputName": "output",
          "blockSize": 256,
          "shrinkSize": 16,
          "scale": 2
        }
      },
      {
        "name": "Anime4K (2x, fastest)",
        "info": "Anime4K CNN (L) upscaler rebuilt as a Core ML model. Very fast, softer than Real-ESRGAN. Anime4K by bloc97 (MIT).",
        "tags": ["2x", "fast", "anime"],
        "type": "multiarray",
        "file": "Anime4K_x2_L.mlmodelc",
        "version": 1,
        "config": {
          "inputName": "input",
          "outputName": "output",
          "blockSize": 256,
          "shrinkSize": 8,
          "scale": 2
        }
      }
    ]
    """

    static let all: [ModelInfo] = {
        do {
            return try JSONDecoder().decode([ModelInfo].self, from: Data(json.utf8))
        } catch {
            LogManager.logger.error("Failed to decode bundled model list: \(error)")
            return []
        }
    }()
}
