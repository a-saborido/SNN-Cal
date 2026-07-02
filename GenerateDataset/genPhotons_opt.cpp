#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TMath.h"
#include "TTree.h"
#include "TVector3.h"


// ── Detector / simulation constants ──────────────────────────────────────────
static constexpr int    nCubletsX      = 10, nCubletsY = 10, nCubletsZ = 10;
static constexpr int    nCellsXY       = 10;
static constexpr int    nCellsZ        = 10;
static constexpr double cellSizeXY     = 3.0;   // mm
static constexpr double cellSizeZ      = 12.0;  // mm
static constexpr double deltaE_vtx_thr = -50e3; // MeV
static constexpr int    n_sensors      = nCellsXY * nCellsZ;               // 100
static constexpr int    nCublets       = nCubletsX * nCubletsY * nCubletsZ; // 1000
static constexpr double lightyield     = 200.0;  // ph/MeV
static constexpr double max_t          = 20.0;   // ns
static constexpr double dt             = 0.2;    // ns
static constexpr int    timesteps      = static_cast<int>(max_t / dt);     // 100

// Flat photon-matrix strides  [nCublets][timesteps][n_sensors]
static constexpr int PM_STRIDE_CUB = timesteps * n_sensors;  // 10 000
static constexpr int PM_STRIDE_T   = n_sensors;               // 100


enum Particle { unclassified = 0, proton = 1, kaon = 2, pion = 3, other = 4 };


// ── ROOT branch pointers (filled by SetBranchAddress) ────────────────────────
static int                  i_evt, n_int;
static std::vector<int>*    pdg        = nullptr;
static std::vector<double>* edep       = nullptr;
static std::vector<double>* deltae     = nullptr;
static std::vector<double>* glob_t     = nullptr;
static std::vector<int>*    cublet_idx = nullptr;
static std::vector<int>*    cell_idx   = nullptr;


// ── Helpers ───────────────────────────────────────────────────────────────────

// Cumulative image-point count up to reflection order n:
//   n=0 → 1,  n=1 → 6,  n=2 → 18, …, n=5 → 102
int total_reflections(int n) {
    int total = 1;   // order 0: direct path only
    for (int i = 1; i <= n; i++)
        total += (i == 1) ? 5 : 4 * (2*i - 1);
    return total;
}

// Returns the flat data and, via n_tot, the N_tot dimension (shape[3]).
std::vector<double> read_matrices(const std::string& filename, int& n_tot) {
    std::ifstream shape_file("shape.txt");
    if (!shape_file) {
        std::cerr << "Failed to open shape.txt\n";
        return {};
    }
    std::vector<size_t> shape;
    {
        std::string line;
        std::getline(shape_file, line);
        std::istringstream iss(line);
        size_t dim;
        while (iss >> dim) shape.push_back(dim);
    }
    if (shape.size() < 4) {
        std::cerr << "shape.txt has fewer than 4 dimensions\n";
        return {};
    }
    n_tot = static_cast<int>(shape[3]);   // N_tot: total stored reflection paths

    size_t total_elements = 1;
    for (size_t d : shape) total_elements *= d;

    std::ifstream binary_file(filename, std::ios::binary);
    if (!binary_file) {
        std::cerr << "Failed to open " << filename << "\n";
        return {};
    }
    std::vector<double> data(total_elements);
    binary_file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(double));
    return data;
}


// ── Solid-angle helpers ───────────────────────────────────────────────────────

double onAxis_SolidAngle(double a, double b, double d) {
    double alpha = a / (2*d);
    double beta  = b / (2*d);
    return 4 * TMath::ASin(alpha*beta / TMath::Sqrt((1 + alpha*alpha)*(1 + beta*beta)));
}

double offAxis_SolidAngle(double A, double B, double a, double b, double d) {
    double sign_A = (A < 0) ? -1.0 : 1.0;  A = std::abs(A);
    double sign_B = (B < 0) ? -1.0 : 1.0;  B = std::abs(B);
    double omega1 = onAxis_SolidAngle(2*(a + sign_A*A), 2*(b + sign_B*B), d);
    double omega2 = onAxis_SolidAngle(2*A,              2*(b + sign_B*B), d);
    double omega3 = onAxis_SolidAngle(2*(a + sign_A*A), 2*B,              d);
    double omega4 = onAxis_SolidAngle(2*A,              2*B,              d);
    return (omega1 - sign_A*omega2 - sign_B*omega3 + sign_A*sign_B*omega4) / 4.0;
}

// Assumes sensors on the upper xz plane.
std::vector<std::vector<std::vector<std::vector<double>>>>
create_matrices(double cellSizeX, double cellSizeY, double cellSizeZ_,
                int nCellsX, int nCellsY, int nCellsZ_) {

    const double n_refr = 2.2;
    const double c = 299.792458 / (n_refr * n_refr);  // mm/ns

    std::vector<std::vector<std::vector<double>>> angle_matrix(
        nCellsX, std::vector<std::vector<double>>(nCellsY, std::vector<double>(nCellsZ_, 0)));
    std::vector<std::vector<std::vector<double>>> time_matrix(
        nCellsX, std::vector<std::vector<double>>(nCellsY, std::vector<double>(nCellsZ_, 0)));

    for (int j = 0; j < nCellsY; j++) {
        TVector3 start(0, j*cellSizeY, 0);
        for (int k = 0; k < nCellsZ_; k++) {
            for (int i = 0; i < nCellsX; i++) {
                TVector3 end(i*cellSizeX, (nCellsY - 0.5)*cellSizeY, k*cellSizeZ_);
                double d = end.Y() - start.Y();
                double A = end.X() - cellSizeX / 2;
                double B = end.Z() - cellSizeZ_ / 2;
                angle_matrix[i][j][k] = offAxis_SolidAngle(A, B, cellSizeX, cellSizeZ_, d);
                time_matrix[i][j][k]  = (end - start).Mag() / c;
            }
        }
    }
    return {angle_matrix, time_matrix};
}


// ── Main processing function ──────────────────────────────────────────────────

void genPhotonTree(const std::string& filename,
                   const std::string& treename,
                   const std::string& outputFilePath,
                   const std::vector<double>& emission_matrix,
                   int max_N,
                   int  n_tot_stored,
                   int  verbose      = 0,
                   bool primary_only = true,
                   int  max_event    = 1000)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    const int total_points = total_reflections(max_N);

    // ── Emission-matrix strides ────────────────────────────────────────────
    //
    // The matrix is stored flat in row-major order with 7 dimensions:
    //
    //   [x_emit][y_emit][z_emit][reflection][x_sensor][z_sensor][{prob, time}]
    //     10      10      10       nRefl5       10        10         2
    //
    // nRefl5 = total_reflections(5) = 102  (always the full depth stored)
    //
    // To reach element [xe][ye][ze][n][sx][sz][v], compute:
    //
    //   index = xe * EM_STR_EX + ye * EM_STR_EY + ze * EM_STR_EZ
    //         +  n * EM_STR_N  + sx * EM_STR_SX + sz * EM_STR_SZ + v
    //
    // Each EM_STR_* is the product of all dimension sizes to its right:
    //
    //   EM_STR_SZ  =  2                         (advance one z-sensor)
    //   EM_STR_SX  =  nCellsZ  * 2   =  20      (advance one x-sensor)
    //   EM_STR_N   =  nCellsXY * 20  = 200      (advance one reflection)
    //   EM_STR_EZ  =  nRefl5   * 200 = 20 400   (advance one emitter-z cell)
    //   EM_STR_EY  =  nCellsZ  * 20 400         (advance one emitter-y cell)
    //   EM_STR_EX  =  nCellsXY * EM_STR_EY      (advance one emitter-x cell)
    //
    // Pre-computing these once avoids re-evaluating the same products in every
    // iteration of the triple sensor loop.

    const int EM_STR_SZ = 2;
    const int EM_STR_SX = nCellsZ       * EM_STR_SZ;
    const int EM_STR_N  = nCellsXY      * EM_STR_SX;
    const int EM_STR_EZ = n_tot_stored  * EM_STR_N;
    const int EM_STR_EY = nCellsZ  * EM_STR_EZ;   // 204 000
    const int EM_STR_EX = nCellsXY * EM_STR_EY;   // 2 040 000
    const double* em    = emission_matrix.data();

    // ── Extract file stem for output naming ───────────────────────────────
    size_t name_start = filename.find_last_of('/');
    size_t name_end   = filename.find_last_of('.');
    std::string name  = filename.substr(name_start + 1, name_end - name_start - 1);

    // ── Open ROOT file / tree ─────────────────────────────────────────────
    TFile* file = TFile::Open(filename.c_str());
    TTree* tree = static_cast<TTree*>(file->Get(treename.c_str())->Clone());

    tree->SetBranchAddress("Tevent_id",              &i_evt);
    tree->SetBranchAddress("Tinteractions_in_event", &n_int);
    tree->SetBranchAddress("Tpdg",                   &pdg);
    tree->SetBranchAddress("Tedep",                  &edep);
    tree->SetBranchAddress("Tdeltae",                &deltae);
    tree->SetBranchAddress("Tglob_t",                &glob_t);
    tree->SetBranchAddress("Tcublet_idx",            &cublet_idx);
    tree->SetBranchAddress("Tcell_idx",              &cell_idx);

    // ── Per-event accumulators ────────────────────────────────────────────
    //
    // photon_matrix is kept flat: [nCublets * timesteps * n_sensors]
    // Access: pm[cub * PM_STRIDE_CUB + t * PM_STRIDE_T + sensor]
    // This replaces the triple nested vector, removing pointer indirection
    // and making the per-event reset a single std::fill call.
    std::vector<int>      photon_matrix(nCublets * PM_STRIDE_CUB, 0);
    std::vector<double>   dEmax(nCublets, 0.0);
    std::vector<double>   Etot(nCublets,  0.0);
    std::vector<TVector3> Ecentroid(nCublets, TVector3(0, 0, 0));
    std::vector<TVector3> sigmaE(nCublets,    TVector3(0, 0, 0));
    std::vector<Particle> p(nCublets, unclassified);
    std::vector<int>      Nint(nCublets,    0);
    std::vector<int>      pdg_max(nCublets, 0);

    std::ofstream outfile;
    if (primary_only) {
        outfile.open(outputFilePath + name + ".dat", std::ios::binary);
        if (!outfile) {
            std::cerr << "Error opening output file for writing!\n";
            return;
        }
    }

    int n_evts = static_cast<int>(tree->GetEntries());
    if (max_event < n_evts) n_evts = max_event;

    // Cached emitter-cell state.
    // The chunk_start index only changes when the emitter cell (x,y,z) changes.
    // Tracking it avoids recomputing three multiplications per interaction and,
    // crucially, removes the vector allocation + copy that the original code
    // performed on every single interaction regardless of whether the cell changed.
    int cached_x = -1, cached_y = -1, cached_z = -1;
    int cached_chunk_start = 0;

    // Per-event random (x, y) shift, uniform over [0, nCellsXY) cell units
    // = [0, nCellsXY * cellSizeXY) mm = one cublet width.
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> shift_dist(0, nCellsXY - 1);
    int shift_x = 0, shift_y = 0;

    // Pre-computed calorimeter extents in global cell units
    const int nGlobalXY = nCubletsX * nCellsXY;   // 100

    // ── Event loop ────────────────────────────────────────────────────────
    for (int i = 0; i < n_evts; i++) {
        if (verbose)
            std::cout << "\tProcessing event " << i << "...\n";

        tree->GetEntry(i);

        // Reset per-event accumulators
        std::fill(photon_matrix.begin(), photon_matrix.end(), 0);
        std::fill(dEmax.begin(),     dEmax.end(),     0.0);
        std::fill(Etot.begin(),      Etot.end(),      0.0);
        std::fill(Ecentroid.begin(), Ecentroid.end(), TVector3(0, 0, 0));
        std::fill(sigmaE.begin(),    sigmaE.end(),    TVector3(0, 0, 0));
        std::fill(p.begin(),         p.end(),         unclassified);
        std::fill(Nint.begin(),      Nint.end(),      0);
        std::fill(pdg_max.begin(),   pdg_max.end(),   0);

        double dE_primary    = 0.0;
        int primary_peak_cub = -1;

        // Invalidate chunk cache at the start of each event
        cached_x = cached_y = cached_z = -1;

        // Sample per-event (x, y) shift in [0, nCellsXY - 1] cell units
        shift_x = shift_dist(rng);
        shift_y = shift_dist(rng);

        // ── Interaction loop ──────────────────────────────────────────────
        for (int j = 0; j < n_int; j++) {
            const double E  = (*edep)[j];
            const double t0 = (*glob_t)[j];
            const double dE = (*deltae)[j];
            if (!(E > 0.0) || t0 >= max_t) continue;

            // ── Decode raw indices, apply per-event (x, y) shift ──────────
            // Cublet z stays the same; cublet x/y and cell-local x/y may change.
            const int raw_cub  = (*cublet_idx)[j];
            const int raw_cell = (*cell_idx)[j];
            const int x_loc =  raw_cell % nCellsXY;
            const int y_loc = (raw_cell % (nCellsXY * nCellsXY)) / nCellsXY;
            const int z_idx =  raw_cell / (nCellsXY * nCellsXY);
            const int gx = (raw_cub % nCubletsX) * nCellsXY + x_loc + shift_x;
            const int gy = ((raw_cub % (nCubletsX * nCubletsY)) / nCubletsX) * nCellsXY + y_loc + shift_y;
            if (gx >= nGlobalXY || gy >= nGlobalXY) continue;   // outside calorimeter
            const int x_idx = gx % nCellsXY;
            const int y_idx = gy % nCellsXY;
            const int cub_i = (raw_cub / (nCubletsX * nCubletsY)) * nCubletsX * nCubletsY
                            + (gy / nCellsXY) * nCubletsX
                            + (gx / nCellsXY);

            // Accumulate energy and interaction count
            Etot[cub_i] += E;
            Nint[cub_i] += 1;

            // Primary vertex: most-negative energy transfer above threshold
            if (dE < deltaE_vtx_thr && dE < dE_primary) {
                primary_peak_cub = cub_i;
                if (verbose > 1)
                    std::cout << "\tPrimary vtx - pdg: " << (*pdg)[j]
                              << " - E: " << E << " - dE: " << dE << ";";
            }

            // Largest energy-loss step → particle identification
            if (dE < dEmax[cub_i]) {
                dEmax[cub_i] = dE;
                const int part_ID = (*pdg)[j];
                pdg_max[cub_i]    = part_ID;
                switch (std::abs(part_ID)) {
                    case 2212: p[cub_i] = proton; break;
                    case  321: p[cub_i] = kaon;   break;
                    case  211: p[cub_i] = pion;   break;
                    default:   p[cub_i] = other;
                }
            }

            // Weighted centroid accumulation
            Ecentroid[cub_i].SetXYZ(Ecentroid[cub_i].X() + x_idx * E,
                                    Ecentroid[cub_i].Y() + y_idx * E,
                                    Ecentroid[cub_i].Z() + z_idx * E);

            // ── Photon distribution via emission matrix ────────────────────
            // Update chunk_start only when the emitter cell changes.
            if (x_idx != cached_x || y_idx != cached_y || z_idx != cached_z) {
                cached_x = x_idx;
                cached_y = y_idx;
                cached_z = z_idx;
                cached_chunk_start = x_idx * EM_STR_EX
                                   + y_idx * EM_STR_EY
                                   + z_idx * EM_STR_EZ;
            }

            const double ph_emitted = E * lightyield;

            // Loop over reflections and sensors.
            // base_n and base_sx hoist the outer-loop multiplications so that
            // the innermost loop (i_sz) only performs a single addition.
            for (int n = 0; n < total_points; n++) {
                const int base_n = cached_chunk_start + n * EM_STR_N;
                for (int i_sx = 0; i_sx < nCellsXY; i_sx++) {
                    const int base_sx = base_n + i_sx * EM_STR_SX;
                    for (int i_sz = 0; i_sz < nCellsZ; i_sz++) {
                        const int   idx       = base_sx + i_sz * EM_STR_SZ;
                        const double prob      = em[idx];
                        const double t_transit = em[idx + 1];
                        const double time     = t0 + t_transit;
                        if (time >= max_t) continue;

                        const int n_photon = static_cast<int>(std::round(ph_emitted * prob));
                        const int step     = static_cast<int>(time / dt);
                        const int sensor_i = i_sz * nCellsXY + i_sx;
                        photon_matrix[cub_i * PM_STRIDE_CUB
                                    + step  * PM_STRIDE_T
                                    + sensor_i] += n_photon;
                    }
                }
            }
        } // end interaction loop

        // Normalise centroid by total energy
        for (int i_cub = 0; i_cub < nCublets; i_cub++) {
            if (!(Etot[i_cub] > 0.0)) continue;
            Ecentroid[i_cub] *= 1.0 / Etot[i_cub];
        }

        // Compute energy dispersion (second pass — apply the same shift)
        for (int j = 0; j < n_int; j++) {
            const int raw_cub  = (*cublet_idx)[j];
            const int raw_cell = (*cell_idx)[j];
            const int x_loc =  raw_cell % nCellsXY;
            const int y_loc = (raw_cell % (nCellsXY * nCellsXY)) / nCellsXY;
            const int z_idx =  raw_cell / (nCellsXY * nCellsXY);
            const int gx = (raw_cub % nCubletsX) * nCellsXY + x_loc + shift_x;
            const int gy = ((raw_cub % (nCubletsX * nCubletsY)) / nCubletsX) * nCellsXY + y_loc + shift_y;
            if (gx >= nGlobalXY || gy >= nGlobalXY) continue;
            const int x_idx = gx % nCellsXY;
            const int y_idx = gy % nCellsXY;
            const int cub_i = (raw_cub / (nCubletsX * nCubletsY)) * nCubletsX * nCubletsY
                            + (gy / nCellsXY) * nCubletsX
                            + (gx / nCellsXY);

            if ((primary_only && cub_i != primary_peak_cub) || !(Etot[cub_i] > 0.0)) continue;

            const double E = (*edep)[j];
            sigmaE[cub_i].SetXYZ(
                sigmaE[cub_i].X() + std::pow(x_idx - Ecentroid[cub_i].X(), 2) * E,
                sigmaE[cub_i].Y() + std::pow(y_idx - Ecentroid[cub_i].Y(), 2) * E,
                sigmaE[cub_i].Z() + std::pow(z_idx - Ecentroid[cub_i].Z(), 2) * E);
        }

        // ── Write event data ──────────────────────────────────────────────
        if (!primary_only) {
            outfile.open(outputFilePath + name + "_" + std::to_string(i) + ".dat",
                         std::ios::binary);
            if (!outfile) {
                std::cerr << "Error opening output file for event " << i << "!\n";
                return;
            }
        }

        for (int i_cub = 0; i_cub < nCublets; i_cub++) {
            if ((primary_only && i_cub != primary_peak_cub) || !(Etot[i_cub] > 0.0)) continue;

            // Sparse photon matrix: only non-zero (t, sensor, count) triples
            const int* pm_cub = photon_matrix.data() + i_cub * PM_STRIDE_CUB;
            for (int i_t = 0; i_t < timesteps; i_t++) {
                for (int i_s = 0; i_s < n_sensors; i_s++) {
                    int entry = pm_cub[i_t * PM_STRIDE_T + i_s];
                    if (entry != 0) {
                        outfile.write(reinterpret_cast<const char*>(&i_t),   sizeof(i_t));
                        outfile.write(reinterpret_cast<const char*>(&i_s),   sizeof(i_s));
                        outfile.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
                    }
                }
            }

            const int stop = 2147483647;
            outfile.write(reinterpret_cast<const char*>(&stop), sizeof(stop));

            outfile.write(reinterpret_cast<const char*>(&i_cub),       sizeof(i_cub));
            outfile.write(reinterpret_cast<const char*>(&Etot[i_cub]), sizeof(Etot[i_cub]));

            double cx = Ecentroid[i_cub].X(), cy = Ecentroid[i_cub].Y(), cz = Ecentroid[i_cub].Z();
            outfile.write(reinterpret_cast<const char*>(&cx), sizeof(cx));
            outfile.write(reinterpret_cast<const char*>(&cy), sizeof(cy));
            outfile.write(reinterpret_cast<const char*>(&cz), sizeof(cz));

            double sX = sigmaE[i_cub].X() / Etot[i_cub];
            double sY = sigmaE[i_cub].Y() / Etot[i_cub];
            double sZ = sigmaE[i_cub].Z() / Etot[i_cub];
            outfile.write(reinterpret_cast<const char*>(&sX), sizeof(sX));
            outfile.write(reinterpret_cast<const char*>(&sY), sizeof(sY));
            outfile.write(reinterpret_cast<const char*>(&sZ), sizeof(sZ));

            outfile.write(reinterpret_cast<const char*>(&Nint[i_cub]), sizeof(Nint[i_cub]));
            outfile.write(reinterpret_cast<const char*>(&p[i_cub]),    sizeof(p[i_cub]));

            if (!primary_only) {
                int is_primary = (i_cub == primary_peak_cub) ? 1 : 0;
                outfile.write(reinterpret_cast<const char*>(&is_primary), sizeof(is_primary));
            }

            if (verbose > 1) {
                std::cout << "\n\ti_cub: " << i_cub
                          << " - Etot: "    << Etot[i_cub]
                          << " - dEmax: "   << dEmax[i_cub]
                          << " - pdg_max: " << pdg_max[i_cub]
                          << " - class: "   << p[i_cub] << "\n";
            }
        }

        if (!primary_only) outfile.close();
    } // end event loop

    if (primary_only) outfile.close();
    file->Close();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    if (verbose)
        std::cout << "Finished. Time taken: " << duration.count() << " s\n";
}


// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string fileName       = "";
    std::string outputFilePath = "./";
    int  verbose      = 0;
    bool primary_only = false;
    int  max_event    = 1000;
    int  reflections  = 0;

    for (int i = 1; i < argc; i++) {
        std::string flag(argv[i]);
        if      (flag.find("fileName")     != std::string::npos) fileName       = flag.substr(11);
        else if (flag == "-f")            { fileName       = argv[++i]; }
        else if (flag.find("output")      != std::string::npos) outputFilePath = flag.substr(9);
        else if (flag == "-o")            { outputFilePath = argv[++i]; }
        else if (flag.find("verbose")     != std::string::npos) verbose        = std::stoi(flag.substr(10));
        else if (flag == "-v")            { verbose        = std::stoi(argv[++i]); }
        else if (flag.find("primary_only")!= std::string::npos || flag == "-po") primary_only = true;
        else if (flag.find("max_event")   != std::string::npos) max_event      = std::stoi(flag.substr(12));
        else if (flag == "-e")            { max_event      = std::stoi(argv[++i]); }
        else if (flag.find("reflections") != std::string::npos) reflections    = std::stoi(flag.substr(14));
        else if (flag == "-r")            { reflections    = std::stoi(argv[++i]); }
    }

    if (fileName.empty()) {
        std::cout << "Missing input file. Use '-f <filename>' or '--fileName=<filename>'\n";
        return 1;
    }
    if (outputFilePath.back() != '/') outputFilePath += '/';

    std::cout << "Reading matrices...\n";
    int n_tot_stored = 0;
    std::vector<double> emission_matrix = read_matrices("emission_matrix_2.bin", n_tot_stored);
    if (emission_matrix.empty()) return 1;

    std::cout << "Matrices loaded. N_tot (stored) = " << n_tot_stored << "\n"
              << "\n---------------------------------------\n\n"
              << "Analyzing file " << fileName << ":\n";

    genPhotonTree(fileName, "outputTree", outputFilePath, emission_matrix,
                  reflections, n_tot_stored, verbose, primary_only, max_event);

    std::cout << "File processing completed.\n";
    return 0;
}
