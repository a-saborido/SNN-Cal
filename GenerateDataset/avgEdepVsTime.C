// ROOT macro: average energy deposition as a function of time
// Loops over all proton/kaon/pion input ROOT files and produces two plots:
//   1. Average energy deposition per bin vs time (MeV, 0.2 ns bins)
//   2. Average cumulative energy fraction vs time (%), i.e. for each timestep
//      the fraction of total event energy deposited by that time, averaged
//      across all events.
// One line per particle type, with consistent colours across both plots.
//
// Usage (from ROOT prompt):
//   .x avgEdepVsTime.C
// or:
//   root -l -b -q avgEdepVsTime.C

void avgEdepVsTime() {

    // --- Configuration ---
    const char* input_base = "/lustre/ific.uv.es/ml/uovi123/snncalo/optCalData";
    const std::vector<std::string> particles = {"proton", "kaon", "pion"};
    const int file_start = 1;
    const int file_end   = 1;
    const char* treename = "outputTree";

    const double max_t = 20.0; // ns
    const double dt    = 0.2;  // ns  (bin width)
    const int    nbins = (int)(max_t / dt);

    // --- Colour palette (one entry per particle, in order) ---
    const std::vector<Color_t> colors = {kBlue+1, kRed+1, kGreen+2};

    // --- Suppress stats box globally ---
    gStyle->SetOptStat(0);

    // --- Per-particle histograms ---
    int npart = (int)particles.size();
    std::vector<TH1D*> hEdep(npart);
    std::vector<TH1D*> hCum(npart);

    for (int ip = 0; ip < npart; ip++) {
        std::string pname = particles[ip];

        hEdep[ip] = new TH1D(("h_edep_" + pname).c_str(),
                             "Average energy deposition vs time;"
                             "Time (ns);"
                             "#LT dE #GT per event (MeV)",
                             nbins, 0.0, max_t);
        hEdep[ip]->Sumw2();

        hCum[ip]  = new TH1D(("h_cum_" + pname).c_str(),
                             "Average cumulative energy fraction vs time;"
                             "Time (ns);"
                             "Cumulative #LT E #GT / #LT E_{tot} #GT (%)",
                             nbins, 0.0, max_t);
        hCum[ip]->Sumw2();
    }

    // --- Tree branch variables ---
    int                  n_int  = 0;
    std::vector<double>* edep   = nullptr;
    std::vector<double>* glob_t = nullptr;

    // Per-event bin buffer (reused each event)
    std::vector<double> evtBinE(nbins, 0.0);

    // Per-particle event counts for normalisation
    std::vector<long long> part_events(npart, 0);
    int files_processed = 0;

    int total_files = npart * (file_end - file_start + 1);
    std::cout << "Starting processing of up to " << total_files << " files..." << std::endl;

    // --- Loop over particle types and file indices ---
    for (int ip = 0; ip < npart; ip++) {
        const std::string& particle = particles[ip];
        std::cout << "\n=== Particle: " << particle << " ===" << std::endl;

        for (int i = file_start; i <= file_end; i++) {

            std::string filepath = std::string(input_base)
                                 + "/" + particle
                                 + "/" + particle
                                 + "_" + std::to_string(i) + ".root";

            std::cout << "  Opening file " << i << "/" << file_end
                      << ": " << filepath << " ..." << std::flush;

            TFile* f = TFile::Open(filepath.c_str(), "READ");
            if (!f || f->IsZombie()) {
                std::cerr << " FAILED" << std::endl;
                if (f) delete f;
                continue;
            }

            TTree* tree = dynamic_cast<TTree*>(f->Get(treename));
            if (!tree) {
                std::cerr << " FAILED (no tree)" << std::endl;
                f->Close();
                delete f;
                continue;
            }

            edep   = nullptr;
            glob_t = nullptr;

            tree->SetBranchAddress("Tinteractions_in_event", &n_int);
            tree->SetBranchAddress("Tedep",                  &edep);
            tree->SetBranchAddress("Tglob_t",                &glob_t);

            Long64_t n_evts = tree->GetEntries();
            std::cout << " OK (" << n_evts << " events)" << std::endl;

            // --- Loop over events ---
            for (Long64_t ievt = 0; ievt < n_evts; ievt++) {
                if (ievt % 100 == 0)
                    std::cout << "    Event " << ievt << "/" << n_evts << "\r" << std::flush;
                tree->GetEntry(ievt);

                std::fill(evtBinE.begin(), evtBinE.end(), 0.0);
                double evtEtot = 0.0;

                // --- Loop over interactions ---
                for (int j = 0; j < n_int; j++) {
                    double E  = (*edep)[j];
                    double t0 = (*glob_t)[j];

                    if (E > 0.0 && t0 < max_t) {
                        hEdep[ip]->Fill(t0, E);

                        int bin = (int)(t0 / dt);
                        if (bin >= nbins) bin = nbins - 1;
                        evtBinE[bin] += E;
                        evtEtot      += E;
                    }
                }

                // Cumulative fraction for this event
                if (evtEtot > 0.0) {
                    double cumsum = 0.0;
                    for (int ibin = 0; ibin < nbins; ibin++) {
                        cumsum += evtBinE[ibin];
                        hCum[ip]->Fill((ibin + 0.5) * dt,
                                       100.0 * cumsum / evtEtot);
                    }
                }
            }

            std::cout << "    Event " << n_evts << "/" << n_evts << " - done" << std::endl;

            part_events[ip] += n_evts;
            files_processed++;

            f->Close();
            delete f;
        }
    }

    std::cout << std::endl;

    long long total_events = 0;
    for (long long n : part_events) total_events += n;

    if (total_events == 0) {
        std::cerr << "No events found. Check input paths." << std::endl;
        return;
    }

    std::cout << "Processed " << files_processed << " files, "
              << total_events << " events total." << std::endl;

    // --- Normalise and style per-particle histograms ---
    std::cout << "Normalising histograms..." << std::endl;
    for (int ip = 0; ip < npart; ip++) {
        if (part_events[ip] == 0) continue;
        hEdep[ip]->Scale(1.0 / (double)part_events[ip]);
        hCum[ip]->Scale(1.0  / (double)part_events[ip]);

        hEdep[ip]->SetLineColor(colors[ip]);
        hEdep[ip]->SetLineWidth(2);
        hCum[ip]->SetLineColor(colors[ip]);
        hCum[ip]->SetLineWidth(2);
    }

    std::cout << "Drawing plots..." << std::endl;

    // =========================================================
    // Plot 1: average energy deposition vs time
    // =========================================================
    TCanvas* c1 = new TCanvas("c_avgEdep", "Average Energy Deposition vs Time", 800, 600);
    c1->SetLeftMargin(0.12);
    c1->SetBottomMargin(0.12);

    // Legend: upper right
    TLegend* leg1 = new TLegend(0.72, 0.72, 0.95, 0.88);
    leg1->SetBorderSize(0);
    leg1->SetFillStyle(0);

    for (int ip = 0; ip < npart; ip++) {
        if (part_events[ip] == 0) continue;
        hEdep[ip]->GetXaxis()->SetTitleSize(0.05);
        hEdep[ip]->GetYaxis()->SetTitleSize(0.05);
        hEdep[ip]->GetYaxis()->SetTitleOffset(1.1);
        hEdep[ip]->GetYaxis()->SetRangeUser(0.0, 8000.0);
        hEdep[ip]->Draw(ip == 0 ? "HIST E" : "HIST E SAME");
        leg1->AddEntry(hEdep[ip], particles[ip].c_str(), "l");
    }
    leg1->Draw();

    c1->SaveAs("avgEdepVsTime.pdf");
    c1->SaveAs("avgEdepVsTime.png");

    // =========================================================
    // Plot 2: average cumulative energy fraction vs time
    // =========================================================
    TCanvas* c2 = new TCanvas("c_cumfrac",
                              "Average Cumulative Energy Fraction vs Time",
                              800, 600);
    c2->SetLeftMargin(0.12);
    c2->SetBottomMargin(0.12);

    // Legend: lower right
    TLegend* leg2 = new TLegend(0.72, 0.18, 0.95, 0.34);
    leg2->SetBorderSize(0);
    leg2->SetFillStyle(0);

    for (int ip = 0; ip < npart; ip++) {
        if (part_events[ip] == 0) continue;
        hCum[ip]->GetXaxis()->SetTitleSize(0.05);
        hCum[ip]->GetYaxis()->SetTitleSize(0.05);
        hCum[ip]->GetYaxis()->SetTitleOffset(1.1);
        hCum[ip]->GetYaxis()->SetRangeUser(0.0, 105.0);
        hCum[ip]->Draw(ip == 0 ? "HIST" : "HIST SAME");
        leg2->AddEntry(hCum[ip], particles[ip].c_str(), "l");
    }
    TLine *line = new TLine(0, 100, 20, 100);
    line->SetLineColor(kBlack);   // black color
    line->SetLineStyle(2);        // dashed
    line->SetLineWidth(2);        // optional, improves visibility
    line->Draw();
    gPad->Update();

    leg2->Draw();

    c2->SaveAs("cumEfracVsTime.pdf");
    c2->SaveAs("cumEfracVsTime.png");

    std::cout << "Plots saved to avgEdepVsTime.{pdf,png} and cumEfracVsTime.{pdf,png}" << std::endl;
}
