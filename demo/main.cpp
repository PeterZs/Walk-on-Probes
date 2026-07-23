#include "core/vector.hpp"
#include "scene/analytical_scene.hpp"
#include "scene/obj_scene.hpp"
#include "solver/hc_solver.hpp"
#include "solver/mvc_solver.hpp"
#include "solver/solver.hpp"
#include "solver/wop_caching_solver.hpp"
#include "solver/wop_solver.hpp"
#include "solver/wos_solver.hpp"
#include "solver/wost_solver.hpp"
#include "stb_image_write.h"
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

int
main(int argc, char* argv[])
{
    CLI::App app{ "WoP Poisson Solver — Walk on Spheres demo" };

    std::string scenePath;
    uint64_t seed = 0;
    int wpp = 256;
    int width = 512;
    int height = 512;
    std::string outputPath = "output.png";
    std::string solverName = "wos";
    bool analytical = false;
    std::string configPath;

    app.add_option("-s,--scene", scenePath, "Path to scene directory");
    app.add_option("-S,--solver", solverName, "Solver: wos | wost | wop | hc | wopc | mvc");
    app.add_flag("--analytical", analytical, "Use analytical source and boundary conditions");
    app.add_option("-r,--seed", seed, "Random seed (0 = use random_device)");
    app.add_option("-w,--wpp", wpp, "Walks per pixel");
    app.add_option("-W,--width", width, "Output image width");
    app.add_option("-H,--height", height, "Output image height");
    app.add_option("-o,--output", outputPath, "Output PNG path");
    app.add_option("--config", configPath, "Path to JSON config file");

    CLI11_PARSE(app, argc, argv);

    // ---- load JSON config if provided ----
    nlohmann::json solverJson;
    nlohmann::json analyticalSettings;

    if (!configPath.empty()) {
        spdlog::info("Loading config from {}", configPath);
        auto j = nlohmann::json::parse(std::ifstream(configPath));

        // Override from JSON if not explicitly set via CLI
        if (app.count("--scene") == 0 && j.contains("scene"))
            scenePath = j["scene"].get<std::string>();
        if (app.count("--solver") == 0 && j.contains("solver") && j["solver"].contains("type"))
            solverName = j["solver"]["type"].get<std::string>();
        if (app.count("--analytical") == 0 && j.contains("analytical"))
            analytical = j["analytical"].get<bool>();
        if (app.count("--seed") == 0 && j.contains("seed"))
            seed = j["seed"].get<uint64_t>();
        if (app.count("--wpp") == 0 && j.contains("wpp"))
            wpp = j["wpp"].get<int>();
        if (app.count("--width") == 0 && j.contains("width"))
            width = j["width"].get<int>();
        if (app.count("--height") == 0 && j.contains("height"))
            height = j["height"].get<int>();
        if (app.count("--output") == 0 && j.contains("output"))
            outputPath = j["output"].get<std::string>();

        if (j.contains("solver"))
            solverJson = j["solver"];
        if (j.contains("analytical_settings"))
            analyticalSettings = j["analytical_settings"];
    }

    // ---- validate required params ----
    if (scenePath.empty()) {
        spdlog::error("Scene path is required (use -s or JSON config)");
        return 1;
    }

    // ---- generate seed if not provided ----
    if (seed == 0) {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    }
    spdlog::info("Seed: {}", seed);

    // ---- load scene ----
    nlohmann::json sceneJson = nlohmann::json::parse(std::ifstream(scenePath + "/config.json"));
    int sceneDim = sceneJson.value("dimension", 2);

    auto run = [&]<int DIM>() -> int {
        std::unique_ptr<WOS::PoissonScene<WOS::Scalar<3>, DIM>> scene;
        if (analytical) {
            auto analyticalScene = std::make_unique<WOS::AnalyticalScene<WOS::Scalar<3>, DIM>>(scenePath);
            if (!analyticalSettings.empty())
                analyticalScene->configure(analyticalSettings);
            spdlog::info("Analytical terms: boundary={}, source={}, screened_kappa={}",
                         analyticalScene->getBoundaryStrength(),
                         analyticalScene->getSourceStrength(),
                         analyticalScene->isScreened() ? analyticalScene->getScreenedKappa() : 0.0);
            scene = std::move(analyticalScene);
        } else
            scene = std::make_unique<WOS::ObjScene<WOS::Scalar<3>, DIM>>(scenePath);

        // ---- create solver ----
        WOS::ThreadRngPool rngPool(seed);
        std::unique_ptr<WOS::Solver<WOS::Scalar<3>, DIM>> solver;
        if (solverName == "wos") {
            solver = std::make_unique<WOS::WoSSolver<WOS::Scalar<3>, DIM>>(*scene, rngPool);
        } else if (solverName == "wost") {
            solver = std::make_unique<WOS::WoStSolver<WOS::Scalar<3>, DIM>>(*scene, rngPool);
        } else if (solverName == "wop") {
            solver = std::make_unique<WOS::WoPSolver<WOS::Scalar<3>, DIM>>(*scene, rngPool);
        } else if (solverName == "hc") {
            solver = std::make_unique<WOS::HCSolver<WOS::Scalar<3>, DIM>>(*scene, rngPool);
        } else if (solverName == "wopc") {
            solver = std::make_unique<WOS::WoPCachingSolver<WOS::Scalar<3>, DIM>>(*scene, rngPool);
        } else if (solverName == "mvc") {
            solver = std::make_unique<WOS::MVCSolver<WOS::Scalar<3>, DIM>>(*scene, rngPool);
        } else {
            solver = std::make_unique<WOS::WoSSolver<WOS::Scalar<3>, DIM>>(*scene, rngPool);
        }

        // Apply solver configuration from JSON or CLI
        if (!solverJson.empty()) {
            solver->configure(solverJson);
        }
        // CLI wpp override
        if (app.count("--wpp") > 0 || solverJson.empty()) {
            if (auto* ws = dynamic_cast<WOS::WoSSolver<WOS::Scalar<3>, DIM>*>(solver.get()))
                ws->setWalksPerPixel(wpp);
            else if (auto* wt = dynamic_cast<WOS::WoStSolver<WOS::Scalar<3>, DIM>*>(solver.get()))
                wt->setWalksPerPixel(wpp);
            else if (auto* wp = dynamic_cast<WOS::WoPSolver<WOS::Scalar<3>, DIM>*>(solver.get()))
                wp->setWalksPerPixel(wpp);
            else if (auto* hc = dynamic_cast<WOS::HCSolver<WOS::Scalar<3>, DIM>*>(solver.get()))
                hc->setWoStWalksPerPixel(wpp);
            else if (auto* wpc = dynamic_cast<WOS::WoPCachingSolver<WOS::Scalar<3>, DIM>*>(solver.get()))
                wpc->setWoStWalksPerPixel(wpp);
            else if (auto* mvc = dynamic_cast<WOS::MVCSolver<WOS::Scalar<3>, DIM>*>(solver.get()))
                mvc->setWoStWalksPerPixel(wpp);
        }

        spdlog::info("Solver: {}", solverName);

        // ---- generate pixel positions ----
        auto [points, pixelIndices] = scene->generateTargetPoints(width, height);

        // ---- solve ----
        spdlog::info("Solving {} pixels ...", points.size());
        std::vector<WOS::Scalar<3>> results;
        solver->solveTimed(points, results);
        const auto& timing = solver->timing();
        spdlog::info("Timing: prepare={:.6f}s, compute={:.6f}s, query={:.6f}s, total={:.6f}s",
                     timing.prepare,
                     timing.compute,
                     timing.query,
                     timing.total);

        // ---- compute relMSE against reference solution ----
        double visualizationMin = 0.0;
        double visualizationMax = 1.0;
        if (analytical) {
            auto* ascene = static_cast<WOS::AnalyticalScene<WOS::Scalar<3>, DIM>*>(scene.get());
            double sumSqErr = 0.0;
            double sumSqExact = 0.0;
            double exactMin = std::numeric_limits<double>::infinity();
            double exactMax = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < results.size(); ++i) {
                auto exact = ascene->exactSolution(points[i]);
                for (int c = 0; c < 3; ++c) {
                    double diff = results[i][c] - exact[c];
                    sumSqErr += diff * diff;
                    sumSqExact += exact[c] * exact[c];
                    exactMin = std::min(exactMin, exact[c]);
                    exactMax = std::max(exactMax, exact[c]);
                }
            }
            double relMSE = sumSqExact > 0.0 ? sumSqErr / sumSqExact : 0.0;
            spdlog::info("relMSE: {:.6e}  (MSE={:.6e}, |exact|^2={:.6e})", relMSE, sumSqErr, sumSqExact);

            if (auto range = ascene->getVisualizationRange()) {
                visualizationMin = range->first;
                visualizationMax = range->second;
            } else if (std::isfinite(exactMin) && std::isfinite(exactMax)) {
                // Zero retains its visual meaning while the exact range prevents
                // analytical settings from being silently clipped to [0, 1].
                visualizationMin = std::min(0.0, exactMin);
                visualizationMax = std::max(0.0, exactMax);
                if (visualizationMax - visualizationMin < 1e-12)
                    visualizationMax = visualizationMin + 1.0;
            }
        } else {
            std::string gtPath = scenePath + "/gt_" + std::to_string(width) + "_" + std::to_string(height) + ".txt";
            std::ifstream gtFile(gtPath);
            if (gtFile.is_open()) {
                int gtW = 0;
                int gtH = 0;
                int gtN = 0;
                bool headerFound = false;
                std::string line;
                while (std::getline(gtFile, line)) {
                    const auto first = line.find_first_not_of(" \t\r\n");
                    if (first == std::string::npos || line[first] == '#')
                        continue;

                    std::istringstream header(line);
                    std::string trailing;
                    if ((header >> gtW >> gtH >> gtN) && !(header >> trailing))
                        headerFound = true;
                    break;
                }

                std::vector<double> gtValues;
                double value = 0.0;
                while (gtFile >> value)
                    gtValues.push_back(value);

                if (!headerFound) {
                    spdlog::error("Invalid GT header in {}", gtPath);
                } else if (gtW != width || gtH != height) {
                    spdlog::error(
                      "GT resolution mismatch in {}: file={}x{}, expected={}x{}", gtPath, gtW, gtH, width, height);
                } else if (gtN != static_cast<int>(results.size())) {
                    spdlog::error(
                      "GT sample count mismatch in {}: file={}, expected={}", gtPath, gtN, results.size());
                } else if (gtValues.size() != results.size() && gtValues.size() != 3 * results.size()) {
                    spdlog::error("GT value count mismatch in {}: got {}, expected {} (scalar) or {} (RGB)",
                                  gtPath,
                                  gtValues.size(),
                                  results.size(),
                                  3 * results.size());
                } else {
                    const bool isRgb = gtValues.size() == 3 * results.size();
                    double sumSqErr = 0.0;
                    double sumSqGt = 0.0;
                    for (std::size_t i = 0; i < results.size(); ++i) {
                        if (isRgb) {
                            for (int c = 0; c < 3; ++c) {
                                double gt = gtValues[3 * i + c];
                                double diff = results[i][c] - gt;
                                sumSqErr += diff * diff;
                                sumSqGt += gt * gt;
                            }
                        } else {
                            double gt = gtValues[i];
                            double diff = results[i][0] - gt;
                            sumSqErr += diff * diff;
                            sumSqGt += gt * gt;
                        }
                    }
                    double relMSE = sumSqGt > 0.0 ? sumSqErr / sumSqGt : 0.0;
                    spdlog::info("relMSE(gt, {}): {:.6e}  (MSE={:.6e}, |gt|^2={:.6e})",
                                 isRgb ? "RGB" : "scalar",
                                 relMSE,
                                 sumSqErr,
                                 sumSqGt);
                }
            } else {
                spdlog::info("No GT file found at {}, skipping relMSE computation against GT", gtPath);
            }
        }

        spdlog::info("Visualization range: [{:.6g}, {:.6g}]", visualizationMin, visualizationMax);

        // ---- map the configured/analytical numeric range to [0, 255] ----
        std::vector<unsigned char> imageData(static_cast<std::size_t>(width) * height * 4, 0);

        for (std::size_t i = 0; i < results.size(); ++i) {
            std::size_t pixelIdx = pixelIndices[i];
            for (int c = 0; c < 3; ++c) {
                double v = (results[i][c] - visualizationMin) / (visualizationMax - visualizationMin);
                v = std::clamp(v, 0.0, 1.0);
                imageData[pixelIdx * 4 + c] = static_cast<unsigned char>(v * 255.0);
            }
            imageData[pixelIdx * 4 + 3] = 255;
        }

        // ---- write output image ----
        std::filesystem::path outputFilePath(outputPath);
        std::filesystem::path outputDirectory = outputFilePath.parent_path();
        if (!outputDirectory.empty()) {
            std::error_code errorCode;
            std::filesystem::create_directories(outputDirectory, errorCode);
            if (errorCode) {
                spdlog::error(
                  "Failed to create output directory {}: {}", outputDirectory.string(), errorCode.message());
                return 1;
            }
        }

        int writeResult = stbi_write_png(
          outputPath.c_str(), width, height, 4, imageData.data(), width * 4 * static_cast<int>(sizeof(unsigned char)));
        if (writeResult == 0) {
            spdlog::error("Failed to write {}", outputPath);
            return 1;
        }

        spdlog::info("Wrote {}", outputPath);

        return 0;
    };

    if (sceneDim == 2)
        return run.template operator()<2>();
    else if (sceneDim == 3)
        return run.template operator()<3>();

    spdlog::error("Unsupported scene dimension: {} (expected 2 or 3)", sceneDim);
    return 1;
}
