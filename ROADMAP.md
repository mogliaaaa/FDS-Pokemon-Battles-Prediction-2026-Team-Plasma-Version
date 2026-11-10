# Roadmap PokePredict

Aggiornata al 28/10/2026. Chiusura competizione: **05/11 23:59** (3 submission finali scelte entro il **04/11**). Consegna report + notebook pubblico: **12/11**.

## Fase 1: EDA (28-29/10)
- [x] Eseguire `notebooks/01_eda.ipynb` e compilare la sezione 8 (conclusioni) con i numeri osservati.
- [x] Documentare l'asimmetria sui KO di P1 e P2 (sezione 5.6), da citare nel report come limite dei dati.
- [ ] Punti da chiarire: Pokémon sotto il livello 100, significato di `null` nelle mosse (switch / KO / bloccato), quanti Pokémon di P2 vengono rivelati in 30 turni.
- [ ] Selezionare 4-6 grafici per il report.

## Fase 2: Feature engineering con conoscenza del gioco (30/10-01/11)
- [ ] Stato al turno 30: KO per lato, HP totali stimati, Pokémon di P2 non rivelati, status attivi.
- [ ] Stima combinata dei Pokémon persi da P2: KO registrati + "spariti" con HP bassi (soglia 0,25 / 0,5 scelta in CV). Tenere le feature P1 e P2 separate, non solo come `diff_*`.
- [ ] Azioni: switch volontari, cambi dopo KO, turni bloccati (sonno / freeze / paralisi).
- [ ] Contatori di sonno e Toxic, boost e mosse di setup, effetti (substitute, leech seed, perish, trapped).
- [ ] Matchup: vantaggio di tipo e velocità tra attivi, copertura di tipo della squadra di P1 contro i Pokémon visti di P2.
- [ ] Danno stimato con la formula Gen 2 (statistiche base + base power + STAB + efficacia).
- [ ] Target encoding delle specie **dentro i fold**.
- [ ] Feature dinamiche: trend HP/KO negli ultimi 5-10 turni, momentum.

## Fase 3: Selezione delle feature (01-02/11)
- [ ] Ablation per gruppi di feature su LightGBM, con verifica sulla LogReg.
- [ ] Permutation importance + curva top-K.
- [ ] Esperimenti da report: PCA/LDA (anche se negativi).

## Fase 4: Modelli (01-03/11)
- [ ] LogReg (regolarizzata), LightGBM, XGBoost, CatBoost, SVM RBF, Random Forest / Extra Trees.
- [ ] Tuning leggero (Optuna o grid) **solo in CV**.
- [ ] Bagging multi-seed, soft voting, stacking con meta-LogReg (OOF predictions).
- [ ] **Stress test di robustezza** (il set finale è privato e diverso dai nostri dati): nei fold di validazione "riparare" l'asimmetria sui KO trasformando in `fnt` una parte dei Pokémon di P2 spariti con pochi HP, e misurare il calo di accuracy. A parità di CV si preferisce il modello più stabile.
- [ ] Ricordare: differenze in CV sotto ~0,5 punti sono rumore; la public LB serve solo come controllo.

## Fase 5: Chiusura (03-05/11)
- [ ] Congelare il codice e scegliere 3 submission diverse: miglior ensemble, miglior modello singolo, variante prudente. Quella prudente usa solo feature robuste (HP residui, KO di P1, status, Pokémon di P2 rivelati), niente che dipenda da come sono registrati i KO di P2.
- [ ] Notebook Kaggle finale: clona il repo, rigenera **esattamente** le 3 submission, cella markdown prima di ognuna.
- [ ] **Selezionare le 3 submission su Kaggle entro il 04/11.**

## Fase 6: Consegna (06-12/11)
- [ ] Report: Feature Engineering & Selection, Model Experimentation & Refinement, Validation Strategy, cosa non ha funzionato e perché.
- [ ] Rendere pubblico il notebook Kaggle; inviare il form entro il 12/11.