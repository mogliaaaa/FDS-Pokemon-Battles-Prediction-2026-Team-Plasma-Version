# Diario degli esperimenti

Registro di tutto quello che proviamo, aggiornato man mano. Ogni esperimento indica:
cosa abbiamo fatto, come lo abbiamo validato, il risultato e la decisione presa.

Dei notebook di submission sono inclusi i tre scelti per la classifica finale (28, 34 e 37, copiati anche in
`final_notebooks/`) e due tappe intermedie (07 e 10); le altre submission citate qui non fanno parte della repo.

**Regole di validazione (valgono per tutti gli esperimenti)**
- Le decisioni si prendono solo con la cross-validation su `train.jsonl` (5-fold stratificata, `shuffle=True`, `random_state=42`).
- Il test set non si usa mai per scegliere feature, modelli o iperparametri.
- La public leaderboard (~30% del test, ±1,2 punti di rumore) serve solo come controllo.

## Riepilogo

| # | Data | Esperimento | Modello | Accuracy CV | Public LB | Decisione |
|---|---|---|---|---|---|---|
| 01 | 26/10/2026 | Baseline semplice: 10 feature sullo stato al turno 30 | Logistic Regression | 0,6622 ± 0,0083 | _da inserire_ | Prima submission |
| 02 | 29/10/2026 | Feature engineering completo: 155 feature in 7 gruppi | Logistic Regression (C=0,1) | 0,6746 ± 0,0083 (CV ripetuta) | - | Riferimento |
| 02 | 29/10/2026 | Feature engineering completo: 155 feature in 7 gruppi | LightGBM (non ottimizzato) | 0,6764 ± 0,0080 (CV ripetuta) | - | Riferimento |
| 02 | 29/10/2026 | Sottoinsieme robusto: 145 feature, senza KO di P2 | LightGBM (non ottimizzato) | 0,6762 ± 0,0086 (CV ripetuta) | - | Base della submission prudente |
| 03 | 29/10/2026 | Target encoding delle specie dentro i fold | LogReg / LightGBM | +0,0002 / 0,0000 (seed 42) | - | Negativo: non usato |
| 04 | 29/10/2026 | Tuning di 17 modelli ammessi + ensemble, tutte le feature | Soft voting (XGB + LGBM + CatBoost + GB) | 0,6800 ± 0,0080 (CV ripetuta) | 0,69200 | Submission 1 (principale) |
| 04 | 29/10/2026 | Come sopra, sottoinsieme robusto (145 feature) | Soft voting (stessi parametri) | 0,6795 ± 0,0078 (CV ripetuta) | 0,69266 | Submission 2 (prudente) |
| 04 | 29/10/2026 | Come sopra, tutte le feature | CatBoost ottimizzato | 0,6774 ± 0,0084 (CV ripetuta), 0,6792 (CV annidata) | 0,68733 | Submission 3 (alternativa) |
| 05 | 30/10/2026 | Analisi degli errori (OOF), 5 feature candidate | LightGBM ottimizzato | 0,6795 (OOF, media di 3 seed) | - | Negativo: nessuna candidata utile, modello calibrato |
| 06 | 30/10/2026 | Feature v2: + 44 feature `team_matchup` (199) | LightGBM ottimizzato (parametri exp 04) | 0,6801 (CV ripetuta), +0,22 su v1 | _da inserire_ | Sotto la soglia di +0,5; provate comunque in submission (10-12) |
| 07 | 30/10/2026 | Feature v3: + 10 feature `hidden_team` (209) | Soft voting / LightGBM (parametri exp 04) | 0,6816 / 0,6774 (5-fold, seed 42) | 0,68866 (main), 0,69200 (prudente) | Neutra: nessun guadagno né in CV né in LB |
| 08 | 30/10/2026 | Selezione delle feature dentro i fold (top-K, filtro di correlazione, gruppi, PCA) | LightGBM ottimizzato / LogReg | 0,6784 con top 80 vs 0,6779 con 155 (CV ripetuta) | - | Neutra: stessa accuracy con metà delle feature, nessun guadagno |
| 09 | 30/10/2026 | Snapshot a t = 22-28 come righe di train in più (P4) e 24 feature `decision_quality` (P1) | LightGBM ottimizzato / LogReg | 0,6786 (snapshot, peso 0,5) / 0,6773 (decisioni) vs 0,6779 (CV ripetuta) | - | Negativo: nessuna delle due supera la soglia |
| 10 | 30/10/2026 | Vincoli monotoni su 9 feature centrali + media su 5 seed (P6), parametri exp 04 | Soft voting robuste (notebook 19) | 0,6801 vs 0,6821 (-0,20, 0/5 fold, seed 42) | _da inserire_ | Negativo: i vincoli tolgono interazioni utili |
| 11 | 31/10/2026 | Forza della squadra: target encoding delle squadre dentro i fold (idea Q1) | LightGBM ottimizzato (parametri exp 04) | 0,6869 vs 0,6779 (+0,90, 14/15 fold, CV ripetuta) | 0,69600 (22), 0,70000 (23), 0,70200 (24) | Guadagno reale; non supera il criterio sulle squadre mai viste: usata in submission, prudente exp 04 tenuta come copertura |
| 12 | 31/10/2026 | Ritaratura di LightGBM, XGBoost, CatBoost e dell'encoder con la forza della squadra | LightGBM / XGBoost / CatBoost | +0,21 / +0,21 / +0,19 sui seed 7 e 123 (miglior candidato) | - | Negativo: si tengono i parametri dell'exp 04 |
| 13 | 01/11/2026 | Encoder con coppie di specie e mascheramento per lato (idee R1 + R3) | LightGBM (parametri exp 04) | 0,6888 vs 0,6867 (+0,21); squadre mai viste +0,43 (P1) e +0,287 (entrambi) sull'encoder attuale | 0,69600 (27), 0,70133 (28), 0,69866 (29) | Non adottato per la regola (manca +0,3 su "entrambi" di 0,013 punti); usato comunque per decisione del team nelle submission 27-29 |
| 14 | 02/11/2026 | Encoder: senza maschera, squadre di P2 rivelate in parte nel pool, risultati residui | LightGBM (parametri exp 04) | Miglior candidato (senza maschera) +0,10, 9/15 fold; +0,04 sui seed nuovi | - | Negativo: l'encoder resta quello di 27-29 |
| 15 | 02/11/2026 | Media dei modelli di 22-24 e 27-29, con e senza LogReg | Media di probabilità | Media dei boosting vs 28: +0,31 e -0,06; LogReg aggiunta alla media: +0,02 e +0,27 | 0,69733 (32) | Negativo per la regola; media + LogReg vs 28 (+0,33 e +0,20) fuori regola, confermata sui seed 4/5/6 (+0,22): submission 32 per decisione del team |
| 16 | 02/11/2026 | Stato delle Spikes, righe specchiate, tipicità delle mosse | LightGBM (parametri exp 04, encoder R1 + R3) | Spikes + mosse 0,6922 vs 0,6888 (+0,34, 10/15 fold); seed nuovi +0,49 (12/15) | _da inserire_ (34-36) | Adottato: Spikes + mosse, unico candidato che passa la regola |

---

## 01 - Baseline semplice (10 feature + Logistic Regression)

**Data:** 26/10/2026
**Notebook:** `simple_submission.ipynb`
**Submission:** `submissions/submission_simple.csv`

### Obiettivo
Fare una prima submission semplice e comprensibile, che usi la timeline della battaglia
(lo starter notebook usa solo le statistiche base delle squadre). Serve anche a verificare che il flusso
dati → feature → validazione → submission funzioni dall'inizio alla fine.

### Feature (10)
Si scorrono i 30 turni osservati e si tengono, per ogni Pokémon, gli ultimi HP e l'ultimo stato visti.
I Pokémon del giocatore 2 mai scesi in campo si considerano a HP pieni (ogni squadra ha 6 Pokémon).

| Feature | Significato |
|---|---|
| `p1_hp_total`, `p2_hp_total` | somma degli HP rimasti (6 = tutti pieni) |
| `hp_diff` | `p1_hp_total - p2_hp_total` |
| `p1_ko`, `p2_ko` | Pokémon esausti (KO) |
| `p1_slp`, `p2_slp` | Pokémon vivi addormentati |
| `p1_par`, `p2_par` | Pokémon vivi paralizzati |
| `p2_seen` | quanti Pokémon del giocatore 2 sono stati visti |

### Modello
`StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)`, parametri di default.

### Risultati
| Validazione | Accuracy |
|---|---|
| 5-fold CV (per fold) | 0,663 / 0,667 / 0,648 / 0,673 / 0,660 |
| **5-fold CV (media)** | **0,6622 ± 0,0083** |
| Public leaderboard | _da inserire dopo la submission_ |

Per confronto:
- lo starter notebook (6 feature sulle statistiche base) fa **0,518** con la stessa CV, praticamente come tirare a caso;
- la regola fissa "vince chi ha più HP al turno 30" fa **0,559**.

**Coefficienti** (sulle feature standardizzate; positivo = aiuta P1 a vincere):

| Feature | Coefficiente |
|---|---|
| `hp_diff` | +0,500 |
| `p2_par` | +0,119 |
| `p1_hp_total` | +0,110 |
| `p2_ko` | +0,067 |
| `p2_slp` | +0,053 |
| `p2_seen` | +0,050 |
| `p1_par` | -0,063 |
| `p1_slp` | -0,103 |
| `p2_hp_total` | -0,409 |
| `p1_ko` | -0,426 |

### Osservazioni
- Lo stato alla fine dei 30 turni contiene molta informazione: si passa dal 51,8% al 66,2% con sole 10 feature.
- Le feature più forti sono la differenza di HP e il numero di KO di P1. Gli stati alterati pesano meno ma nella direzione attesa: è un male avere Pokémon propri addormentati o paralizzati, un bene se li ha l'avversario.
- La varianza tra i fold è bassa (±0,8 punti), quindi la stima è stabile.

### Decisione e prossimi passi
- Usata come **prima submission** di controllo.
- Prossimi passi: più feature sull'andamento della battaglia (momentum degli HP, mosse usate, vantaggio di tipo) e modelli non lineari (gradient boosting).

---

## 02 - Feature engineering completo e ablation per gruppi

**Data:** 29/10/2026
**Notebook:** [`notebooks/03_features.ipynb`](notebooks/03_features.ipynb) (preprocessing in [`notebooks/02_preproc.ipynb`](notebooks/02_preproc.ipynb))
**Codice:** `src/preprocessing.py`, `src/move_damage.py`, `src/features.py`, `src/validation.py`

### Obiettivo
Costruire un insieme di feature completo a partire da EDA e preprocessing, e misurare quanto è utile ogni gruppo,
con due modelli di riferimento non ottimizzati.

### Feature (155 numeriche, 7 gruppi)
| Gruppo | N. | Contenuto |
|---|---|---|
| `static` | 22 | statistiche della squadra di P1, livelli, strumenti, statistiche del lead di P2 |
| `state` | 21 | stato dopo 30 turni: KO, Pokémon visti / spariti, HP totali, status, stima dei Pokémon persi da P2 (`lost_est`) |
| `trend` | 12 | differenza di HP totali ai turni 10/20/25, pendenza negli ultimi 5/10 turni, min/max/media, tempi dei KO |
| `action` | 48 | switch, turni bloccati, categorie di mosse, gruppi di mosse (sonno, Toxic, setup, Spikes, ...), danno atteso |
| `status_effect` | 23 | turni passati con ogni status ed effetto (Substitute, Leech Seed, confusione, ...) |
| `boost` | 20 | boost del Pokémon attivo al turno 30 e boost massimo raggiunto, per statistica |
| `matchup` | 9 | vantaggio di tipo e velocità effettiva (boost, paralisi) degli attivi al turno 30, copertura di tipo delle squadre |

Il **sottoinsieme robusto** (145 feature) toglie le 10 feature basate sui KO registrati di P2 o sui suoi Pokémon spariti
(EDA, sezione 5.6).

### Modelli
- LogReg: imputazione con mediana + `StandardScaler` + L2 (C=0,1).
- LightGBM: 400 alberi, learning rate 0,03, 15 foglie, `min_child_samples=40`, subsample 0,8, colsample 0,7.

### Risultati
CV ripetuta (seed 42, 7, 123 × 5 fold):

| Configurazione | LogReg | LightGBM |
|---|---|---|
| tutte le feature | 0,6746 ± 0,0083 | 0,6764 ± 0,0080 |
| sottoinsieme robusto | 0,6734 ± 0,0078 | 0,6762 ± 0,0086 |

Ablation (seed 42, media dei 5 fold):

| Configurazione | LogReg | LightGBM |
|---|---|---|
| tutte le feature | 0,6770 | 0,6781 |
| solo state / tutte - state | 0,6621 / 0,6664 | 0,6547 / 0,6728 |
| solo trend / tutte - trend | 0,6526 / 0,6788 | 0,6477 / 0,6799 |
| solo action / tutte - action | 0,5923 / 0,6720 | 0,6001 / 0,6727 |
| solo boost / tutte - boost | 0,5388 / 0,6710 | 0,5357 / 0,6709 |
| solo matchup / tutte - matchup | 0,5728 / 0,6755 | 0,5704 / 0,6745 |
| solo static / tutte - static | 0,5120 / 0,6746 | 0,5488 / 0,6767 |
| solo status_effect / tutte - status_effect | 0,5386 / 0,6767 | 0,5319 / 0,6778 |

Rispetto alla baseline 01 (0,6622): **+1,2 / +1,4 punti**.

### Osservazioni
- Lo **stato della battaglia** è il nucleo: da solo arriva a 0,66, e toglierlo costa più di tutto.
- **Boost e azioni** contano più di quanto dicano le correlazioni (toglierli costa 0,5-0,7 punti): agiscono tramite interazioni.
- `trend` è **ridondante** con `state`: da solo è forte, ma toglierlo non peggiora (+0,2, nel rumore).
- Il **sottoinsieme robusto non perde nulla**: i KO registrati di P2 non aggiungono informazione. La submission prudente non costa accuracy.
- Le 63 battaglie con Pokémon sotto-livellati hanno livelli bassi **da entrambe le parti** (`p1_min_level` e `p2_lead_level` correlati a 0,996): probabilmente un formato diverso. Tenute e segnalate.
- Tutte le configurazioni stanno tra 0,675 e 0,68, con differenze per lo più dentro il rumore (std ≈ 0,8 punti).

### Decisione e prossimi passi
- Tenere tutti i gruppi per ora; provare a togliere `trend` nella selezione delle feature.
- Prossimi passi: permutation importance e curva top-K, tuning in CV, altri modelli (CatBoost, SVM, ensemble), stress test sui KO di P2.

---

## 03 - Target encoding delle specie (negativo)

**Data:** 29/10/2026
**Notebook:** [`notebooks/03_features.ipynb`](notebooks/03_features.ipynb), sezione 5

### Obiettivo
Verificare se la specie del lead di P2 e dei Pokémon attivi al turno 30 aggiunge informazione.

### Metodo
`TargetEncoder` (binario, con cross-fitting interno) dentro un `ColumnTransformer` nella pipeline, quindi stimato solo
sulla parte di training di ogni fold. Le specie sconosciute ricevono la media globale.

### Risultati
| Modello | Senza specie | Con target encoding | Differenza |
|---|---|---|---|
| LogReg | 0,6770 | 0,6772 | +0,0002 |
| LightGBM | 0,6781 | 0,6781 | 0,0000 |

### Decisione
Non usato. È coerente con l'EDA: i win rate per specie stanno tra 0,47 e 0,57.

---

## 04 - Confronto e tuning di tutti i modelli ammessi, ensemble, stress test

**Data:** 29/10/2026
**Notebook:** [`notebooks/04_models.ipynb`](notebooks/04_models.ipynb), eseguito sul PC potente (circa 1,5 ore in tutto)
**Codice:** `src/models.py`, `src/validation.py`
**Submission:** `05_submission_main`, `06_submission_prudent`, [`07_submission_alternative`](notebooks/07_submission_alternative.ipynb)

### Obiettivo
Confrontare tutti i modelli ammessi dalle regole sulle 155 feature dell'exp 02, ottimizzarne gli iperparametri,
provare gli ensemble e misurare la sensibilità alla registrazione dei KO di P2.

### Metodo
- 17 modelli, ognuno in una `Pipeline` (imputazione e scaling dentro i fold): LogReg, Ridge, SVM lineare, LDA, QDA,
  Gaussian NB, k-NN, SVM RBF, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, HistGB,
  LightGBM, XGBoost, CatBoost.
- Tuning: `RandomizedSearchCV` (15-60 candidati) o `GridSearchCV`, 5-fold, seed 42, accuracy.
- Valutazione dei parametri migliori con CV ripetuta (seed 42, 7, 123) su tutte le feature e sul sottoinsieme robusto.
- **Stress test**: si addestra sui fold normali e si valida su fold in cui il 50% / 100% dei Pokémon di P2 spariti con
  HP ≤ 0,5 diventano KO registrati. "Calo" = accuracy pulita - accuracy al 100%, in punti.
- Ensemble dei 4 migliori: soft voting e stacking (meta-modello LogReg).
- CV annidata (esterna 5-fold seed 7, ricerca completa dentro ogni fold) per i 3 migliori.

### Risultati

| Modello | Default | Tuning (seed 42) | CV ripetuta, tutte | CV ripetuta, robusto | Calo stress | CV annidata |
|---|---|---|---|---|---|---|
| **Soft voting** | - | - | **0,6800 ± 0,0080** | **0,6795 ± 0,0078** | 0,00 | - |
| Stacking | - | - | 0,6795 ± 0,0081 | 0,6787 ± 0,0077 | 0,01 | - |
| XGBoost | 0,6557 | 0,6828 | 0,6791 ± 0,0078 | 0,6752 ± 0,0074 | 0,12 | 0,6751 |
| LightGBM | 0,6713 | 0,6798 | 0,6779 ± 0,0097 | 0,6782 ± 0,0088 | 0,04 | 0,6781 |
| **CatBoost** | 0,6787 | 0,6809 | 0,6774 ± 0,0084 | 0,6765 ± 0,0081 | -0,15 | **0,6792** |
| Gradient Boosting | 0,6761 | 0,6785 | 0,6773 ± 0,0090 | 0,6769 ± 0,0077 | 0,12 | - |
| SVM RBF | 0,6736 | 0,6795 | 0,6770 ± 0,0107 | 0,6742 ± 0,0083 | 0,74 | - |
| HistGB | 0,6675 | 0,6783 | 0,6765 ± 0,0086 | 0,6748 ± 0,0090 | 0,12 | - |
| SVM lineare | 0,6778 | 0,6778 | 0,6760 ± 0,0086 | 0,6734 ± 0,0070 | 0,70 | - |
| Ridge | 0,6773 | 0,6777 | 0,6752 ± 0,0087 | 0,6738 ± 0,0073 | 0,55 | - |
| LDA | 0,6774 | 0,6780 | 0,6752 ± 0,0089 | 0,6739 ± 0,0074 | 0,63 | - |
| LogReg | 0,6764 | 0,6778 | 0,6750 ± 0,0098 | 0,6732 ± 0,0075 | 0,57 | - |
| AdaBoost | 0,5740 | 0,6731 | 0,6727 ± 0,0074 | 0,6731 ± 0,0072 | -0,22 | - |
| Random Forest | 0,6651 | 0,6696 | 0,6666 ± 0,0080 | 0,6654 ± 0,0088 | -0,14 | - |
| QDA | fallisce | 0,6673 | 0,6649 ± 0,0091 | 0,6607 ± 0,0107 | 0,38 | - |
| Extra Trees | 0,6618 | 0,6668 | 0,6647 ± 0,0075 | 0,6646 ± 0,0081 | -0,05 | - |
| k-NN | 0,6008 | 0,6602 | 0,6594 ± 0,0071 | 0,6536 ± 0,0063 | 0,46 | - |
| Decision Tree | 0,5752 | 0,6531 | 0,6520 ± 0,0071 | 0,6520 ± 0,0071 | 0,00 | - |
| Gaussian NB | 0,6398 | 0,6442 | 0,6444 ± 0,0088 | 0,6384 ± 0,0106 | 0,25 | - |

Parametri migliori dei membri del soft voting:
- XGBoost: 720 alberi, lr 0,0121, max_depth 6, min_child_weight 1,34, subsample 0,60, colsample 0,69, gamma 0,0105, lambda 3,54;
- LightGBM: 1.424 alberi, lr 0,0066, 12 foglie, min_child_samples 106, subsample 0,66, colsample 0,63, alpha 0,61, lambda 2,71;
- CatBoost: 763 iterazioni, lr 0,039, depth 5, l2_leaf_reg 1,15, random_strength 1,36, bagging_temperature 0,15;
- Gradient Boosting: 466 alberi, lr 0,048, max_depth 3, min_samples_leaf 91, subsample 0,93.

### Osservazioni
- **I primi 12 modelli stanno in mezzo punto** (0,675-0,680), sotto la soglia di rumore. Rispetto alla LightGBM non
  ottimizzata dell'exp 02 il tuning guadagna 0,2-0,4 punti: il limite adesso sono le feature, non il modello.
- **Il punteggio del tuning è ottimistico.** XGBoost scende da 0,6828 (tuning) a 0,6791 (CV ripetuta) e a 0,6751
  (CV annidata): la ricerca si era adattata ai fold con seed 42. CatBoost e LightGBM invece reggono.
  Per il report vanno usati i numeri della CV annidata.
- **Gli ensemble aggiungono poco**: soft voting +0,1 punti sul miglior modello singolo (rumore), ma con varianza
  un po' più bassa. Lo stacking costa 41 minuti e fa peggio del soft voting.
- **Stress test**: i modelli lineari, le SVM e LDA perdono 0,55-0,74 punti, quindi si appoggiano sul conteggio dei KO
  di P2. I boosting perdono 0-0,12 punti: usano soprattutto HP e `lost_est`. Un modello lineare non è adatto come
  submission prudente.
- **Il sottoinsieme robusto costa poco** (soft voting -0,05, LightGBM +0,03). Fa eccezione XGBoost (-0,4).
- QDA con i default fallisce (covarianza singolare per le feature collineari); funziona con `reg_param=0,8`.
  I default di AdaBoost (alberi senza limite di profondità) e Decision Tree vanno in overfitting.
- Alcuni parametri sono al bordo dello spazio di ricerca (k-NN `n_neighbors=301`, NB `var_smoothing=1`, LightGBM
  lr minimo e alberi vicino al massimo): allargare lo spazio darebbe al massimo qualche decimo.

### Decisione
Tre submission, ognuna con un notebook autonomo (dati grezzi → preprocessing → feature → modello → CSV):
1. **Principale**: soft voting dei 4 boosting, tutte le feature.
2. **Prudente**: stesso soft voting, sottoinsieme robusto (senza feature sui KO di P2).
3. **Alternativa**: CatBoost singolo, tutte le feature (miglior CV annidata, robusto allo stress test).

Scartati come submission: XGBoost singolo (tuning ottimistico), stacking (costoso, nessun guadagno), modelli lineari
(sensibili ai KO di P2). Prossimo passo per superare 0,68: feature nuove, non altro tuning.

---

## 05 - Analisi degli errori (negativo per le feature candidate)

**Data:** 30/10/2026
**Notebook:** [`notebooks/08_error_analysis.ipynb`](notebooks/08_error_analysis.ipynb), eseguito sul PC del team
**Modello:** LightGBM ottimizzato dell'exp 04, tutte le 155 feature

### Obiettivo
Capire dove sbaglia il modello per decidere quali feature costruire: confidenza e calibrazione, accuracy per tipo
di battaglia, errori "sicuri", segnale rimasto nei residui (feature esistenti, specie, 5 feature candidate),
learning curve.

### Metodo
Predizioni fuori fold (OOF): Stratified 5-fold ripetuta con seed 42, 7, 123, media delle tre probabilità. Solo train.

### Risultati
OOF accuracy: 0,6798 / 0,6784 / 0,6755 per seed, **0,6795** con la media dei tre (coerente con l'exp 04).

| Fascia di confidenza | Quota di battaglie | Accuracy | Quota degli errori |
|---|---|---|---|
| p fra 0,35 e 0,65 | 46% | 0,53-0,62 | **62%** |
| p fra 0,2 e 0,35 o fra 0,65 e 0,8 | 35% | 0,68-0,77 | 30% |
| p ≤ 0,2 o ≥ 0,8 | 19% | 0,85-0,95 | 8% |

| Tipo di battaglia | Quota | Accuracy |
|---|---|---|
| \|`diff_hp_total`\| < 0,5 (equilibrate) | 34% | 0,619 |
| stesso `lost_est` per i due lati | 31% | 0,620 |
| nessun KO registrato | 24% | 0,648 |
| \|`diff_hp_total`\| ≥ 1,5 (già decise) | 22% | 0,800 |
| boost, sonno, Explosion, hazard, Substitute, P2 con Pokémon non visti | - | tutte entro ±1,5 punti dalla media |

Feature candidate (correlazione di Spearman con il residuo `y - p` / massima correlazione con le feature esistenti):

| Candidata | r con il residuo | max \|r\| con le esistenti |
|---|---|---|
| HP pesati per la forza della specie (diff) | -0,062 | 0,97 |
| Pokémon vivi addormentati o congelati (diff) | -0,004 | 0,64 |
| cambi di vantaggio negli HP | -0,011 | 0,29 |
| lato dell'ultimo KO / turni dall'ultimo KO | -0,004 / -0,008 | 0,50 / 0,75 |
| mosse diverse rivelate (diff) | 0,007 | 0,20 |

Learning curve (validazione): 0,659 con 1.200 battaglie, 0,675 con 5.733, **0,680 con 8.000**; train 0,76.

### Osservazioni
- **Il modello è calibrato**: la curva di calibrazione sta sulla diagonale e la probabilità media coincide con la
  vittoria reale in ogni fascia di `diff_hp_total`, `diff_lost_est`, `p1_ko` e del trend. Le probabilità sono
  quelle giuste per l'informazione disponibile.
- **Gli errori stanno nelle battaglie ancora aperte al turno 30**: è incertezza vera sul resto della battaglia (che
  dura 30-80 turni), non un difetto del modello.
- **Gli errori più sicuri sono rimonte o rumore**: in due P1 vince con 5 KO contro quasi nessuno di P2; in un caso
  P1 vince con 6 KO registrati (probabile abbandono o etichetta sbagliata).
- **Specie**: 2 coppie specie-colonna su 80 con |z| > 3, contro circa 0,2 attese per caso, e su 57 e 151 battaglie:
  non utilizzabile. Conferma l'exp 03.
- **Feature candidate**: gli HP pesati per specie sono ridondanti (la correlazione con il residuo è lo stesso effetto
  delle feature HP esistenti, vedi sotto); le altre hanno |r| ≤ 0,018. Nessuna promettente.
- Le feature HP esistenti hanno r ≈ -0,06 con il residuo, ma è un effetto della correlazione di Spearman su un
  residuo di un target binario, non informazione persa: la calibrazione per fasce è perfetta.
- **La learning curve sale ancora** (circa +0,3 punti ogni 1.000 battaglie): più dati aiuterebbero, ma i dati
  esterni sono vietati. Spiega in parte perché i modelli finali, addestrati su 10.000 battaglie, fanno ~0,69 sulla
  public leaderboard.
- **Errore di metodo** nella prima versione della sezione 4 (errori sicuri): il cambio di segno delle feature in
  base al vincitore predetto produceva differenze spurie sulle feature statiche. Riscritta confrontando errori e
  predizioni corrette dentro la stessa classe predetta; va rieseguita.

### Decisione
Le feature candidate non si costruiscono. Con l'informazione dei primi 30 turni siamo vicini al limite.
Resta un ultimo esperimento sulle feature (06): confronto di tipo e velocità fra **tutti** i Pokémon vivi di P1 e
quelli già visti di P2, non solo fra i due attivi. Si tiene solo se la CV ripetuta guadagna più di ~0,5 punti;
altrimenti le feature si congelano e si passa a notebook finale e report.

---

## 06 - Feature v2: confronto fra le squadre ancora in gioco (`team_matchup`)

**Data:** 30/10/2026
**Notebook:** [`notebooks/09_features_v2.ipynb`](notebooks/09_features_v2.ipynb)
**Codice:** `src/features.py` (`team_matchup_features`; `build_features(..., team_matchup=True)`)
**Submission:** [`10_submission_v2_main`](notebooks/10_submission_v2_main.ipynb), `11_submission_v2_prudent`, `12_submission_v2_lgbm`

### Obiettivo
Dare al modello informazioni sulle squadre **ancora in gioco** al turno 30, non solo sui due Pokémon attivi, per
ridurre gli errori sulle rimonte (indizi dall'exp 05: boost offensivi e copertura di tipo).

### Feature (44, quattro sottogruppi)
Per ogni battaglia: elenco dei Pokémon ancora in gioco (P1: tutti e 6 tranne i KO; P2: visti, non KO e non spariti
con HP ≤ 0,5, cioè la stessa stima di `lost_est`). Ogni Pokémon "attacca" tutti quelli avversari con le mosse
offensive rivelate (tipo e categoria dal dataset; Hidden Power esclusa) o, se non ne ha rivelate, con i propri tipi.
- `tm_types_damage` (14): copertura, bersagli superefficaci, attaccanti non resistiti, `sweep` (miglior attaccante
  contro il bersaglio peggiore), `pressure` (danno pesato per gli HP avversari);
- `tm_speed` (6): velocità massima rimasta (boost e paralisi), quanti sono più veloci del più veloce avversario;
- `tm_status_hp` (13): HP dei vivi divisi per stato (il peso lo stima il modello);
- `tm_other` (11): vivi con mosse di cura, con attacchi prioritari o Counter/Mirror Coat, switch volontari nei
  turni 21-30, BST massimo dei Pokémon di P1 mai entrati.

Fonti: solo la battaglia, la tabella delle specie del train e la type chart. Tutte e 44 restano nel sottoinsieme robusto.
Regola fissata prima dell'esecuzione: tenere il gruppo solo con un guadagno > 0,5 punti.

### Risultati
CV appaiata (stessi fold, seed 42, 7, 123 × 5):

| Modello | Feature | v1 → v2 | Guadagno (punti) | Fold migliorati |
|---|---|---|---|---|
| LightGBM | tutte | 0,6779 → 0,6801 | +0,22 ± 0,52 | 10/15 |
| LightGBM | robuste | 0,6782 → 0,6793 | +0,12 ± 0,61 | 11/15 |
| LogReg | tutte | 0,6750 → 0,6733 | -0,17 ± 0,41 | 5/15 |
| LogReg | robuste | 0,6732 → 0,6726 | -0,06 ± 0,42 | 5/15 |

Ablation (LightGBM): ogni sottogruppo aggiunto da solo alle v1 va da -0,10 a +0,01; tolto dalle v2 costa da -0,08 a
-0,27. Stress test: calo 0,10 punti (v1: 0,04). Il gruppo prende il 18% del gain di LightGBM (`diff_tm_hp_healthy`
3ª feature; `pressure` e `sweep` fra le prime 20). Correlazione di Pearson con il residuo del modello v1:
`sweep` e `pressure` circa 0,03 (sopra il rumore, ~0,02). Attacchi prioritari e Counter: quasi sempre 0.

### Osservazioni
- Le feature portano un po' di informazione nuova (`sweep`, `pressure`), ma il modello le usa soprattutto al posto
  di HP e `lost_est`: il guadagno netto resta nel rumore.
- L'ablation è incoerente (nessun sottogruppo aiuta da solo, ognuno "costa" se tolto): tipico del rumore.
- La LogReg peggiora leggermente (44 colonne in più, molto correlate fra loro).

### Decisione
Sotto la soglia: le v1 restano il riferimento per la scelta finale. Su decisione del team le v2 si provano comunque
in submission (le submission giornaliere non costano nulla e il set finale è diverso): soft voting con tutte le
199 feature, soft voting con le 189 robuste, LightGBM con 199. Parametri dell'exp 04, non ri-ottimizzati sulle v2.

---

## 07 - Squadra nascosta di P2 (`hidden_team`)

**Data:** 30/10/2026
**Codice:** `src/features.py` (`fit_team_cooccurrence`, `hidden_team_features`; `build_features(..., team_model=...)`)
**Submission:** `13_submission_v3_main`, `14_submission_v3_prudent`, `15_submission_v3_lgbm`

### Idea
Di P2 vediamo solo i Pokémon entrati in campo (in circa metà delle battaglie ne restano di nascosti al turno 30). Le
10.000 squadre **complete** di P1 del train dicono quali specie vengono usate insieme. Dati i Pokémon di P2 rivelati,
per ogni specie si stima la probabilità di essere fra quelli nascosti (media delle probabilità condizionate
P(specie | specie rivelata), escluse le specie già viste, normalizzata). Si usano solo i nomi delle specie, mai l'esito.

### Feature (10)
`p2_hid_exp_bst`, `p2_hid_exp_total_bst` (forza attesa dei nascosti, per Pokémon e in totale), `p2_hid_exp_spe`,
`p2_hid_n_faster` (nascosti probabilmente più veloci del più veloce di P1 ancora in gioco), `p2_hid_threat` /
`p1_hid_cover` / `diff_hid_cover_threat` (efficacia di tipo attesa fra nascosti di P2 e Pokémon di P1 ancora in gioco),
`p1_strength_est` / `p2_strength_est` / `diff_strength_est` (somma di HP × BST di ciò che resta, nascosti inclusi).
Tutte restano nel sottoinsieme robusto (P2 "in gioco" = non KO e non sparito con HP ≤ 0,5).

### Validazione
Per decisione del team provata direttamente in submission, senza un esperimento di CV separato: la CV 5-fold (seed 42)
è nella sezione 6 dei notebook 13-15. La co-occorrenza è stimata su tutte le squadre del train anche dentro quella CV;
non usa l'etichetta, quindi non c'è leakage del target.

### Risultati
| Notebook | Modello | Feature | CV 5-fold (seed 42) | Public LB |
|---|---|---|---|---|
| 13 | soft voting | v3, 209 | 0,6816 ± 0,0072 | 0,68866 |
| 14 | soft voting | v3 robuste, 199 | 0,6786 ± 0,0043 | 0,69200 |
| 15 | LightGBM | v3, 209 | 0,6774 ± 0,0051 | - |

Confronto con le v1 (stesso seed 42): LightGBM 0,6798 → 0,6774 (-0,24 punti). Public LB delle submission v1: 0,69200 (main),
0,69266 (prudente). Le differenze sono tutte nel rumore (CV ±0,5 punti per seed; LB ±1,2 punti, 0,33 punti = 5 battaglie).

### Decisione
Neutra: la squadra nascosta stimata dalla co-occorrenza non aggiunge informazione utile, coerente con l'exp 05 (le specie
non portano segnale). Le v1 restano il riferimento per la scelta finale.

---

## 08 - Selezione delle feature (neutra)

**Data:** 30/10/2026
**Notebook:** [`notebooks/16_feature_selection.ipynb`](notebooks/16_feature_selection.ipynb)
**Codice:** `src/selection.py` (`CorrelationFilter`, `top_k_selector`, `with_selection`)

### Obiettivo
Gli exp 05-07 hanno aggiunto feature senza guadagno. Qui si prova il contrario: **togliere** feature. La learning curve
dell'exp 05 mostrava un divario train / validazione ampio (0,76 vs 0,68); se dipendesse da feature ridondanti o
rumorose, toglierle dovrebbe aiutare.

### Metodo
Feature v1 (155), LightGBM ottimizzato (parametri exp 04) e LogReg, CV ripetuta appaiata (stessi 15 fold della
baseline). Ogni selezione è il primo passo della `Pipeline`, quindi viene ristimata sul fold di training:
- **top-K**: un LightGBM veloce tiene le K feature con più gain (K = 10…120);
- **filtro di correlazione**: gruppi con |Spearman r| ≥ soglia (complete linkage), si tiene il medoide (0,95 / 0,9 / 0,8);
- **gruppi interi**: senza `static`, senza `trend`, senza entrambi;
- **PCA** (imputazione → scaling → PCA → modello), per completezza.

Regola fissata prima: adottare se guadagna ≥ 0,3 punti, oppure se perde ≤ 0,1 punti con al massimo il 60% delle feature.

### Risultati
| Configurazione | Feature | LightGBM | LogReg |
|---|---|---|---|
| tutte (v1) | 155 | 0,6779 | 0,6750 |
| top 120 / 80 / 60 | 120 / 80 / 60 | -0,07 / **+0,05** / -0,05 | -0,18 / -0,26 / -0,57 |
| top 40 / 20 / 10 | 40 / 20 / 10 | -0,33 / -1,50 / -1,98 | -0,93 / -1,33 / -1,42 |
| filtro 0,95 / 0,9 / 0,8 | ~152 / 148 / 137 | -0,14 / 0,00 / -0,07 | +0,02 / +0,01 / -0,01 |
| senza `static` / `trend` / entrambi | 133 / 143 / 121 | -0,01 / -0,18 / -0,36 | -0,03 / -0,12 / -0,16 |
| PCA 20 / 60 (LightGBM), 10-80 (LogReg) | | -1,66 / -1,20 | da -2,74 a -0,31 |

Guadagni in punti rispetto a tutte le feature. Top 80 con LightGBM: +0,05 ± 0,25, migliora in 10 fold su 15.
Stabilità (K = 80): 59 feature scelte in tutti i 15 fold. Stress test sui KO di P2: calo 0,16 punti (tutte le
feature: 0,04).

### Osservazioni
- Le feature sono poco ridondanti: a |r| ≥ 0,9 se ne tolgono solo 7 (es. `p2_ko` / `p2_first_ko_turn`,
  `p1_hp_total` / `p1_lost_est`).
- Con LightGBM l'accuracy resta piatta fino a 60-80 feature e poi crolla; la LogReg perde con qualsiasi taglio.
- Le statistiche medie della squadra (`static`) vengono scelte in tutti i fold, ma toglierle non costa nulla: le feature
  continue ricevono un po' di gain negli alberi anche senza segnale. L'importanza per gain non misura l'utilità.
- PCA sempre peggio: le direzioni con più varianza non sono quelle che predicono il vincitore.
- Il divario train / validazione non dipende da feature in eccesso: il limite è l'informazione nei 30 turni.

### Decisione
Per la regola, top 80 con LightGBM verrebbe adottato: stessa accuracy con circa metà delle feature. Il guadagno però
è nel rumore, quindi è una semplificazione, non un miglioramento. I modelli con tutte le feature restano il
riferimento. Se si prova in submission, la selezione va dentro la pipeline e ristimata sul train completo.


---

## 09 - Snapshot come righe di train e qualità delle decisioni (negativo)

**Data:** 30/10/2026
**Notebook:** [`notebooks/17_snapshots_decisions.ipynb`](notebooks/17_snapshots_decisions.ipynb)
**Codice:** `src/features.py` (`build_features_at`, `decision_quality_features`), `src/preprocessing.py`
(`truncate_timeline`, `pokemon_states(..., n_turns)`), `src/validation.py` (`augmented_cv_scores`)

### Obiettivo
Provare le due idee con la priorità più alta (vedi "Idee da provare"):
- **P4**: la learning curve sale ancora, quindi si aggiungono righe di train prese dalle stesse battaglie tagliate a
  t = 22/24/26/28 (stessa etichetta, `sample_weight` minore, colonna `snapshot_turn`);
- **P1**: feature che descrivono il giocatore invece dello stato della battaglia.

### Metodo
- Numero di turni come parametro (`n_turns`, default 30). Controllo: a 30 turni le feature v1 ricostruite sono
  identiche a quelle del notebook 03 (differenza massima 0).
- Le righe delle snapshot vanno **solo nel train di ogni fold**, prese dalle battaglie di quel fold; la validazione usa
  sempre le righe a t = 30. Gli stessi fold di `cv_scores`, quindi il confronto è appaiato.
- 24 feature `decision_quality` (P1, P2, diff): efficacia media degli attacchi contro il Pokémon colpito, quota di
  superefficaci e di resistiti / immuni, attacchi su un'immunità, mosse di stato sprecate (bersaglio già con status o
  immune per la type chart), efficacia dell'attacco avversario sul Pokémon entrato con uno switch volontario. Hidden
  Power escluso. Nessuna usa i KO: tutte restano nel sottoinsieme robusto.
- Regola fissata prima: adottare con guadagno ≥ +0,5 punti e almeno 11 fold su 15 migliorati.

### Risultati
| Configurazione | Modello | Accuracy | Guadagno (punti) | Fold migliorati |
|---|---|---|---|---|
| v1 (riferimento) | LightGBM | 0,6779 | - | - |
| v1 + decisioni | LightGBM | 0,6773 | -0,06 ± 0,57 | 7/15 |
| snapshot 26-28, peso 0,5 | LightGBM | 0,6768 | -0,11 ± 0,50 | 6/15 |
| snapshot 22-28, peso 0,5 | LightGBM | 0,6786 | +0,07 ± 0,59 | 10/15 |
| snapshot 22-28, peso 1 | LightGBM | 0,6774 | -0,05 ± 0,68 | 4/15 |
| snapshot 22-28, peso (t - 20) / 10 | LightGBM | 0,6781 | +0,02 ± 0,52 | 6/15 |
| snapshot 22-28, peso 0,5 + decisioni | LightGBM | 0,6761 | -0,18 ± 0,45 | 5/15 |
| v1 + decisioni | LogReg | 0,6734 | -0,16 ± 0,45 | 2/15 |
| snapshot 22-28, peso 0,5 | LogReg | 0,6737 | -0,13 ± 0,36 | 5/15 |

Stress test (seed 42): calo 0,13 punti con le snapshot (v1: 0,04). La correlazione di `diff_hp_total` con il vincitore
cresce con il turno (0,28 a t = 22, 0,37 a t = 30).

### Osservazioni
- Le feature sulle decisioni hanno un po' di segnale (|r| fino a 0,10 con il vincitore, fino a 0,04 con il residuo del
  modello v1, sopra il rumore di ~0,02), ma non cambiano le predizioni: come le `team_matchup` dell'exp 06.
- Le snapshot sono molto correlate con la riga a t = 30 della stessa battaglia: aggiungono righe, non informazione.
  La learning curve che sale indica che servirebbero **battaglie nuove**, non copie di quelle che abbiamo.
- Con questo esperimento nove tentativi di feature e rappresentazioni sono finiti nel rumore: il limite sembra
  l'informazione contenuta nei primi 30 turni.

### Decisione
Negativo: nessuna configurazione supera la regola. Le v1 con i modelli dell'exp 04 restano il riferimento. Il codice
resta in `src/` (disattivato di default).

---

## 10 - Vincoli monotoni + media su 5 seed (negativo)

**Data:** 30/10/2026
**Notebook:** `18`, `19`, `20`
**Codice:** `src/models.py` (`MONOTONE_DIRECTIONS`, `monotone_params`, `SeedBagging`, `seed_bagged`)

Idea P6 (vedi "Idee da provare"): vincoli monotoni su 9 feature con direzione fissata dalla logica del gioco (HP rimasti,
Pokémon persi) per LightGBM, XGBoost e CatBoost, e media delle probabilità su 5 seed (42-46). Parametri dell'exp 04.

| Notebook | Modello | CV 5-fold (seed 42) | Modello exp 04, stessi fold | Guadagno |
|---|---|---|---|---|
| 19 | soft voting, 145 robuste | 0,6801 ± 0,0061 | 0,6821 | -0,20 punti, 0/5 fold |

Il calo è piccolo ma sistematico: i vincoli tolgono interazioni non monotone che il modello usava davvero (per esempio
molti HP rimasti ma su un Pokémon addormentato). La media sui seed da sola non è stata provata separatamente.
**Decisione:** negativo. Nota: la prudente dell'exp 04 fa 0,6821 sul seed 42, il miglior valore di CV fra le submission.

---

## 11 - Forza della squadra (target encoding delle squadre)

**Data:** 31/10/2026
**Notebook:** [`notebooks/21_team_strength.ipynb`](notebooks/21_team_strength.ipynb); submission 22-24
**Codice:** `src/team_strength.py` (`TeamStrengthEncoder`, `with_team_strength`, `mask_teams`),
`src/features.py` (`team_key_features`, `build_features(..., team_keys=True)`)

### Obiettivo
Idea Q1 (vedi "Idee da provare"). Le squadre si ripetono fra le battaglie: il tasso di vittoria di una
squadra, stimato sul train, è informazione che lo stato al turno 30 non contiene.

### Metodo
- Chiavi: le 6 specie di P1 e le specie di P2 rivelate nei 30 turni.
- `TeamStrengthEncoder`, primo passo della Pipeline: `te_p1` (tasso di vittoria smussato, K = 10, della squadra
  esatta di P1), `te_p2` (tasso delle squadre del train compatibili con i Pokémon rivelati di P2), conteggi (log),
  `diff_te`. Le squadre di P2 rivelate per intero entrano nel pool con etichetta 1 - y.
- Niente leakage: dentro il fold di train ogni riga è codificata dagli altri fold interni (cross-fitting, 5 fold);
  validazione e test usano tutto il train del fold. Squadra mai vista → 0,5.
- Regola fissata prima: guadagno ≥ +0,5 punti con ≥ 11/15 fold, nessun guadagno con chiavi mescolate, con il 100%
  di squadre mai viste al massimo -0,1 punti sotto la baseline.

### Risultati
| Configurazione (LightGBM salvo dove indicato) | Accuracy | Guadagno (punti) | Fold migliorati |
|---|---|---|---|
| v1 (riferimento) | 0,6779 | - | - |
| v1 + forza della squadra | 0,6869 | +0,90 ± 0,65 | 14/15 |
| robuste + forza della squadra (vs robuste 0,6782) | 0,6859 | +0,77 ± 0,50 | 15/15 |
| v1 + squadra, K = 30 | 0,6868 | +0,89 | 15/15 |
| v1 + squadra, senza squadre di P2 nel pool | 0,6839 | +0,60 | 11/15 |
| LogReg v1 + squadra (vs 0,6750) | 0,6801 | +0,51 | 13/15 |
| Controllo: chiavi mescolate | 0,6773 | -0,06 | 7/15 |

CV raggruppata per squadra di P1 (seed 42): +0,36 punti, 4/5 fold; **non** è un dato puro su squadre mai viste: circa il 60% delle righe di validazione riceve comunque un `te_p1` informativo, perché la squadra può entrare nel pool come squadra di P2. Stress test sui KO di P2: nessun calo.

Submission (CV 5-fold seed 42, confronto appaiato col modello dell'exp 04 sugli stessi fold):

| Notebook | Modello | Encoder | CV | Modello exp 04 | Guadagno | Public LB |
|---|---|---|---|---|---|---|
| 22 | soft voting, 155 + squadra | `train_unseen=0.3` | 0,6884 ± 0,0060 | 0,6812 | +0,72, 4/5 fold | 0,69600 |
| 23 | soft voting, 145 robuste + squadra | `counts=False, train_unseen=0.5` | 0,6880 ± 0,0050 | 0,6821 | +0,59, 5/5 fold | 0,70000 |
| 24 | LightGBM, 155 + squadra | `train_unseen=0.3` | 0,6905 ± 0,0045 | 0,6798 | +1,07, 5/5 fold | 0,70200 |

Squadre mai viste in validazione (guadagno in punti):

| Encoder | Pulito | 60% mai viste | 100% mai viste |
|---|---|---|---|
| default | +0,90 | -0,15 | -0,59 |
| `counts=False` | +0,64 | +0,09 | -0,30 |
| `train_unseen=0.3` | +0,88 | +0,26 | -0,23 |
| `counts=False, train_unseen=0.3` | +0,75 | +0,21 | -0,24 |
| `counts=False, train_unseen=0.5` | +0,67 | +0,26 | -0,11 |

Public LB rispetto alle submission dell'exp 04: soft voting 0,69200 → 0,69600, soft voting robuste 0,69266 → 0,70000. Fra le tre submission 22-24 ci sono 0,6 punti (9 battaglie su ~1.500), dentro il rumore della leaderboard (±1,2): la public conferma la direzione del guadagno, non stabilisce quale delle tre sia la migliore.

### Osservazioni
- Il primo esperimento con un guadagno chiaro: tutti i precedenti stavano entro ±0,2 punti. Il segnale viene dalle
  squadre (sparisce mescolando le chiavi) ma non abbiamo ancora una misura pulita su squadre di P1 davvero mai viste (vedi la CV raggruppata sopra).
- **Correzione:** gran parte del segnale è rarità, non forza. Una squadra di P1 che non compare altrove vince il 40,4% delle battaglie (2.732), contro circa il 54% delle altre; fra le squadre frequenti i tassi di vittoria di due metà delle battaglie non correlano (0,006 su 89 squadre, campione piccolo). La causa non è stabilita. **Correzione (exp 14):** il dato 0,006 non si riproduce: mediando su 20 divisioni casuali la correlazione è 0,43 (10-19 battaglie, 49 squadre), 0,18 (20-49, 48 squadre), 0,32 (50-99, 10 squadre), 0,48 (100 o più, 7 squadre). Oltre alla rarità c'è quindi una differenza reale fra squadre, su campioni piccoli.
- Con l'encoder di default il modello impara che "0,5 con 0 battaglie" vuol dire squadra rara, e peggiora se tutte
  le squadre sono nuove. Segnare una parte delle righe di train come mai viste (come un dropout) riduce il danno senza
  perdere il guadagno.

### Decisione
Per la regola non è adottata come riferimento: la variante più sicura manca il criterio di 0,01 punti (nel rumore).
Il guadagno è grande dove le squadre sono condivise (il test condivide circa il 70% delle squadre di P1 col train);
il rischio è un set privato con squadre diverse. Decisione: usarla nelle submission 22 (soft voting, `train_unseen=0.3`),
23 (soft voting robuste, `counts=False, train_unseen=0.5`) e 24 (LightGBM, `train_unseen=0.3`), e tenere la prudente
dell'exp 04 fra le tre finali come copertura.

---

## 12 - Ritaratura con la forza della squadra (negativo)

**Data:** 31/10/2026
**Notebook:** [`notebooks/25_tuning_team_strength.ipynb`](notebooks/25_tuning_team_strength.ipynb)

### Obiettivo
I parametri delle submission 22-24 vengono dall'exp 04, tarati senza la forza della squadra. Ritarare solo i modelli
di quelle submission (LightGBM, XGBoost, CatBoost; il Gradient Boosting resta com'è) e, per LightGBM, anche l'encoder.

### Metodo
`RandomizedSearchCV` (LightGBM 40 candidati, XGBoost 30, CatBoost 20) sui 5 fold del seed 42, in uno spazio stretto
attorno ai valori attuali; encoder dentro la pipeline. Verifica dei 3 migliori candidati di ogni modello sui 10 fold
dei seed 7 e 123, mai visti dalla ricerca, in confronto appaiato con i parametri attuali. Regola fissata prima:
adottare con guadagno ≥ +0,3 punti e ≥ 7/10 fold migliorati.

### Risultati
| Modello | Ricerca (seed 42): migliore / attuali | Verifica (seed 7, 123): attuali | Miglior candidato in verifica | Fold migliorati |
|---|---|---|---|---|
| LightGBM | 0,6903 / 0,6905 | 0,6848 | +0,21 punti (il 1° della ricerca: -0,55) | 6/10 |
| XGBoost | 0,6908 / 0,6883 | 0,6856 | +0,21 | 5/10 |
| CatBoost | 0,6914 / 0,6849 | 0,6873 | +0,19 | 6/10 |

### Osservazioni
- Il miglior candidato di LightGBM sul seed 42 è il peggiore fuori campione (-0,55, 1 fold su 10): lo stesso ottimismo
  della ricerca visto nell'exp 04. La verifica su seed separati lo ha intercettato.
- I parametri dell'exp 04 erano già vicini all'ottimo anche con le nuove feature.

### Decisione
Negativo: nessun modello supera la regola. Le submission 22-24 restano con i parametri dell'exp 04 e l'encoder
`train_unseen=0.3`.

---

## 13 - Encoder con coppie di specie e mascheramento per lato (non adottato, per poco)

**Data:** 01/11/2026
**Notebook:** [`notebooks/26_encoder_unseen_teams.ipynb`](notebooks/26_encoder_unseen_teams.ipynb)
**Codice:** `src/team_strength.py` (`pairs=True`, `per_side=True`, `unseen_cv_scores`), spenti di default

### Obiettivo
Idee R1 e R3, in una sola esecuzione: ridurre il danno sulle squadre mai viste e averne una misura
pulita. Le squadre di validazione vengono tolte dal pool dell'encoder, in entrambi i ruoli, dopo l'addestramento. Le
specie restano note. Limiti: il modello si addestra comunque su quelle battaglie; specie nuove non sono coperte.

### Regola fissata prima
Contro l'encoder attuale (`train_unseen=0.3`), LightGBM, 15 fold appaiati: CV normale al massimo 0,1 punti sotto; in entrambi
gli scenari con squadre mai viste almeno +0,3 punti e almeno 11/15 fold; chiavi mescolate sotto +0,2 punti.

### Risultati
| Encoder | CV normale | Squadre mai viste, solo P1 | Squadre mai viste, entrambi |
|---|---|---|---|
| Senza squadra | 0,6779 | 0,6779 | 0,6779 |
| Attuale (22, 24) | 0,6867 | 0,6745 | 0,6743 |
| R3 per lato | 0,6845 | 0,6760 | 0,6742 |
| R1 coppie | 0,6900 | 0,6763 | 0,6752 |
| **R1 + R3** | **0,6888** | **0,6788** | **0,6771** |
| Encoder del 23 (su LightGBM, tutte le feature) | 0,6846 | 0,6769 | 0,6767 |

R1 + R3 sull'encoder attuale: +0,21 (8/15), +0,43 (15/15), +0,287 (12/15). Sul modello senza squadra: +1,09, +0,09, -0,08.
Chiavi mescolate: -0,29 sul modello senza squadra. Etichette invertite per le 195 battaglie di una squadra: punteggi di coppia
invariati.

### Osservazioni
- Misura pulita sulle squadre mai viste: l'encoder attuale perde 0,34-0,36 punti rispetto al modello senza squadra, più del
  -0,23 stimato con la maschera dell'exp 11.
- Il guadagno sulle squadre mai viste viene dalle due modifiche insieme: da sole portano poco, e R3 da sola perde sulla CV
  normale (-0,22, 2/15). Sono confronti secondari sugli stessi fold, quindi lievemente ottimisti.

### Decisione
Per la regola **non adottato**: "entrambi" manca +0,3 di 0,013 punti, molto meno del rumore della misura, ma la regola era
fissata prima e il risultato si riporta come tale. Le submission 22-24 restano valide.

**Decisione del team, contro la regola:** usare comunque R1 + R3 nelle submission 27-29, perché non è peggio in nessuno scenario e le submission giornaliere non costano nulla. È la seconda eccezione dopo l'exp 11, va dichiarata nel report. La public non può confermare la scelta: le squadre del test sono in buona parte note e il rumore è ±1,2 punti.

Submission 27-29 (5 fold, solo seed 42, quindi più rumorosi dei 15 fold sopra):

| Notebook | Modello | CV normale | Modello exp 04, stessi fold | Mai viste, solo P1 | Mai viste, entrambi | Public LB |
|---|---|---|---|---|---|---|
| 27 | LightGBM, 155 | 0,6896 ± 0,0044 | 0,6798 | 0,6783 (-0,15) | 0,6772 (-0,26) | 0,69600 |
| 28 | Soft voting, 155 | 0,6891 ± 0,0032 | 0,6812 | 0,6830 (+0,18) | 0,6795 (-0,17) | 0,70133 |
| 29 | Soft voting, 145 robuste | 0,6874 ± 0,0038 | 0,6821 | 0,6806 (-0,15) | 0,6788 (-0,33) | 0,69866 |

Fra parentesi il confronto col modello dell'exp 04 senza squadra, sugli stessi fold. Sul seed 42 LightGBM risulta meno neutro che
sui 15 fold (+0,09 e -0,08): il seed 42 è favorevole al modello senza squadra (0,6798 contro 0,6779 sui 15 fold). Sulla CV normale le
tre submission sono equivalenti alle corrispondenti 22-24 (0,6884, 0,6880, 0,6905).

---

## 14 - Rifinitura dell'encoder (negativo)

**Data:** 02/11/2026
**Notebook:** [`notebooks/30_encoder_refinements.ipynb`](notebooks/30_encoder_refinements.ipynb)
**Codice:** `src/team_strength.py` (`soft_p2=True`, `residual=True`, `refit_pool`), spenti di default

### Obiettivo
Stime di squadra migliori, dopo il nuovo obiettivo di 0,71 in public. Parte dell'obiettivo è fuori dal nostro controllo: da 0,702 a 0,71
sono 12 battaglie su circa 1.500, con rumore ±1,2 punti.

### Controlli descrittivi sul train, prima dell'esperimento
| Battaglie della squadra | Squadre | Battaglie | Vittorie | Correlazione fra due metà (media su 20 divisioni) |
|---|---|---|---|---|
| 1 | 3.130 | 3.130 | 0,421 | - |
| 2 | 460 | 920 | 0,546 | 0,218 |
| 3-4 | 276 | 924 | 0,536 | 0,093 |
| 5-9 | 187 | 1.219 | 0,559 | 0,226 |
| 10-19 | 49 | 644 | 0,548 | 0,425 |
| 20-49 | 48 | 1.422 | 0,541 | 0,179 |
| 50-99 | 10 | 668 | 0,497 | 0,317 |
| 100 o più | 7 | 1.073 | 0,513 | 0,478 |

Strumenti, livelli e ordine degli slot non aggiungono identità: dentro uno stesso insieme di specie, chi² per grado di libertà
0,93 (strumenti e livelli, 45 insiemi) e 1,05 (ordine, 111 insiemi). Le chiavi restano le specie. L'ordine degli slot è quasi
sempre diverso (9.236 chiavi ordinate su 10.000 battaglie).

### Regola fissata prima
Contro R1 + R3, LightGBM, 15 fold appaiati, per il candidato con la CV normale migliore: almeno +0,3 punti e almeno 10/15 fold; almeno +0,15
punti su seed nuovi (1, 2, 3); chiavi mescolate sotto +0,2 punti sul modello senza squadra e al massimo 0,2 punti sotto R1 + R3 con le stesse
chiavi. Candidati: senza maschera (`pairs=True`, `train_unseen=0`), B (`soft_p2`, idea R2), C (`residual`). Nessuna iterazione.

**Regola cambiata prima di eseguire.** La versione iniziale aveva i candidati `n_inner=20`, B, C,
A + B, A + B + C e chiedeva di non perdere più di 0,1 punti sulle squadre mai viste. Ora quel dato è solo descrittivo, perché il set finale
segue la distribuzione del test; da dichiarare nel report.

### Risultati
| Encoder | CV normale | Contro R1 + R3 | Fold | Mai viste, entrambi (descrittivo) | Chiavi mescolate, sul modello senza squadra |
|---|---|---|---|---|---|
| Senza squadra | 0,6779 | - | - | 0,6779 | - |
| R1 + R3 (riferimento) | 0,6888 | - | - | 0,6771 | -0,29 |
| Senza maschera | 0,6898 | +0,10 | 9/15 | 0,6737 (-0,34) | -0,26 |
| B `soft_p2` | 0,6870 | -0,18 | 4/15 | 0,6739 (-0,32) | -0,39 |
| C `residual` | 0,6883 | -0,05 | 6/15 | 0,6761 (-0,10) | -0,30 |

Seed nuovi, candidato senza maschera: +0,04 punti, 7/15 fold. Etichetta invertita di una battaglia: le sue colonne di squadra non cambiano per
riferimento, senza maschera e B; con C il massimo è 0,0016.

### Osservazioni
- B alza la correlazione di `diff_te` con l'esito sulle righe di train (da 0,1245 a 0,1417), ma l'accuracy scende: la correlazione non bastava
  come evidenza.
- Togliere la maschera non dà guadagno misurabile e costa 0,34 punti sulle squadre mai viste, come lasciava prevedere l'exp 11.

### Decisione
**Non adottato**: nessun candidato raggiunge +0,3 punti. L'encoder resta R1 + R3.

---

## 15 - Media di modelli con encoder diversi (negativo per la regola)

**Data:** 02/11/2026
**Notebook:** [`notebooks/31_blending.ipynb`](notebooks/31_blending.ipynb)
**Codice:** `src/validation.py` (`oof_proba`, `fold_accuracy`)

### Obiettivo
Prima opzione fra le varianti dell'exp 14: i modelli delle submission hanno modelli, feature ed encoder diversi, e una media può ridurre gli errori. Abbiamo
scelto di provare anche LogReg, che da sola è più debole ma è l'unico membro lineare; il suo contributo non era
misurato. Il notebook confronta anche LightGBM e soft voting sugli stessi 15 fold.

### Metodo
Pesi fissati prima e mai tarati. Media semplice dei sei modelli di 22, 23, 24, 27, 28, 29, tutti e sei perché nessuno sia scelto in base alla public.
Con LogReg: 75% media dei boosting, 25% LogReg (C dell'exp 04, encoder R1 + R3, tutte le feature).

### Regola fissata prima
Confronti appaiati, 15 fold sui seed 42/7/123 e 15 fold sui seed nuovi 1/2/3. Media dei boosting contro submission 28: almeno +0,15 punti su entrambi.
Media + LogReg contro media dei boosting: almeno +0,15 punti su entrambi, altrimenti LogReg resta fuori. La soglia è sotto lo 0,5 abituale perché pesi
uguali non scelgono nulla; resta un guadagno piccolo, vicino al rumore della misura.

### Risultati
| Modello | Seed 42/7/123 | Seed 1/2/3 | Media dei 30 fold |
|---|---|---|---|
| 22 soft voting | 0,6891 | 0,6844 | 0,6868 |
| 23 soft voting robuste | 0,6851 | 0,6832 | 0,6842 |
| 24 LightGBM | 0,6867 | 0,6827 | 0,6847 |
| 27 LightGBM R1 + R3 | 0,6888 | 0,6863 | 0,6876 |
| 28 soft voting R1 + R3 | 0,6874 | 0,6873 | 0,6874 |
| 29 soft voting robuste R1 + R3 | 0,6862 | 0,6874 | 0,6868 |
| LogReg R1 + R3 | 0,6843 | 0,6849 | 0,6846 |
| Media dei boosting | 0,6905 | 0,6866 | 0,6886 |
| Media + LogReg | 0,6907 | 0,6893 | 0,6900 |

Media dei boosting contro 28: +0,31 (11/15) e -0,06 (6/15). LogReg aggiunta alla media: +0,02 e +0,27.

### Osservazioni
- **Confronto fuori regola:** media + LogReg contro 28 dà +0,33 (11/15) e +0,20 (10/15), l'unico risultato positivo su entrambi i gruppi di seed.
  È stato guardato dopo i risultati, quindi vale come ipotesi. I seed 1/2/3 non sono più nuovi e tutti i fold ricampionano le stesse 10.000
  battaglie: l'errore di un confronto resta dell'ordine di 0,2-0,3 punti (stima).
- **LogReg sbaglia in modo diverso:** è in disaccordo coi boosting sul 10,5-11,8% delle battaglie; i boosting fra loro sul 3,4-7,0%.
- **Soft voting o LightGBM:** con R1 + R3 si equivalgono (LightGBM +0,14 e -0,09); con l'encoder di 22 e 24 è meglio il soft voting (+0,25 e +0,17).
- **Submission 24:** ha la public migliore (0,702), ma il suo 0,6905 sul seed 42 era favorevole: sui 30 fold fa 0,6847, 0,29 punti sotto la 27.

### Decisione
Per la regola **nessuna delle due medie è adottata** e LogReg resta fuori.

**Decisione del team, terza eccezione alla regola (dopo exp 11 ed exp 13):** preparare comunque la submission media + LogReg (notebook 32,
`ProbaBlend` in `src/models.py`). Regola di conferma fissata prima di eseguire: sui seed nuovi 4, 5, 6 almeno +0,15 punti contro la submission 28 sugli
stessi 15 fold; altrimenti non entra fra le tre finali. Va dichiarata nel report.

| Notebook | Modello | Seed 4/5/6 | Contro 28 | Public LB |
|---|---|---|---|---|
| 32 | 75% media dei sei boosting, 25% LogReg | 0,6880 | +0,22, 9/15 fold (28: 0,6858) | 0,69733 |

**Conferma superata**: +0,22 punti contro la soglia di +0,15. Sui seed 4/5/6 la media batte anche gli altri membri: 22 +0,13 (8/15), 27 +0,24 (10/15), 24 +0,33 (10/15), 29 +0,34 (12/15), LogReg +0,51 (11/15), 23 +0,56 (12/15). Il notebook riproduce il notebook 31 (28 sul seed 42: 0,6891). Limite: i tre gruppi di seed (+0,33, +0,20, +0,22) ricampionano le stesse 10.000 battaglie, quindi non sono tre prove indipendenti. Sul test la media differisce dalla 28 nel 3,5% delle battaglie.

---

## 16 - Spikes, righe specchiate, tipicità delle mosse (Spikes + mosse adottate)

**Data:** 02/11/2026
**Notebook:** [`notebooks/33_spikes_mirror_moves.ipynb`](notebooks/33_spikes_mirror_moves.ipynb)
**Codice:** `src/features.py` (`hazard_features`, `move_key_features`), `src/models.py` (`MirrorEstimator`, `mirror_frame`), `src/move_typicality.py`

### Obiettivo
Misurare su LightGBM le tre idee migliori dell'ultima lista (02/11), finora provate solo su modelli veloci. Riferimento: il modello della submission 27.

### Regola fissata prima
Per ogni candidato (Spikes, specchio, mosse, Spikes + specchio, Spikes + mosse), in confronto appaiato col riferimento: sui seed 42/7/123 almeno +0,3 punti e almeno 10/15 fold; sui seed nuovi 8/9/10 almeno +0,15 punti;
colonne nuove mescolate fra le battaglie sotto +0,2 punti. Nessuna iterazione: chi passa diventa una submission.

### Risultati
| Candidato | Seed 42/7/123 | Contro il riferimento | Fold migliori / peggiori | Seed nuovi 8/9/10 |
|---|---|---|---|---|
| Riferimento (submission 27) | 0,6888 | - | - | 0,6848 |
| S Spikes | 0,6892 | +0,04 | 8 / 7 | - |
| M specchio | 0,6894 | +0,06 | 8 / 7 | - |
| L mosse | 0,6890 | +0,02 | 9 / 6 | - |
| S + M | 0,6914 | +0,26 | 10 / 4 | - |
| **S + L** | **0,6922** | **+0,34** | **10 / 4** | **0,6897 (+0,49, 12/15)** |

Colonne mescolate: S -0,13, L -0,19 punti. Specchio: due applicazioni restituiscono i valori originali (127 colonne con controparte, 39 senza).

### Osservazioni
- **I modelli veloci sopravvalutavano le idee singole:** le Spikes da sole danno +0,04 su LightGBM contro +0,65 su LogReg. LightGBM poteva già approssimare lo stato dai conteggi.
- **Il guadagno compare solo in combinazione** (+0,04 e +0,02 da sole, +0,34 e +0,49 insieme). Non ne abbiamo una spiegazione stabilita; aggiungere colonne cambia anche quelle campionate da LightGBM, quindi una parte può essere rumore.
- Tutti i fold ricampionano le stesse 10.000 battaglie: i seed nuovi non sono dati indipendenti.
- Nelle battaglie equilibrate (`|diff_hp_total|` ≤ 0,5) P1 vince il 69,3% se solo P2 ha le Spikes sul campo e il 49,3% se solo P1. Il dato descrittivo non separa l'effetto delle Spikes dall'abilità del giocatore.

### Decisione
**Adottate Spikes + mosse** per la regola fissata prima. Submission 34 (LightGBM), 35 (soft voting) e 36 (media con LogReg). Il notebook 33 ha misurato solo LightGBM: 35 e 36 hanno una regola di conferma propria, almeno +0,15 punti sui seed 8, 9, 10 contro lo stesso modello senza le colonne nuove.

| Notebook | Modello | CV seed 8/9/10 | Contro lo stesso modello senza colonne nuove | Public LB |
|---|---|---|---|---|
| 34 | LightGBM | 0,6897 (seed 42/7/123: 0,6922) | +0,49, 12/15 fold (seed 42/7/123: +0,34, 10/15) | _da inserire_ |
| 35 | Soft voting | 0,6895 | +0,41, 13/15 fold: confermata | _da inserire_ |
| 36 | Media dei sei boosting + LogReg | 0,6919 | +0,41, 13/15 fold: confermata | _da inserire_ |

La 34 riproduce il notebook 33 (0,6922 e 0,6888) e il controllo sulla 28 torna (seed 42: 0,6891). Le predizioni cambiano sul 5,8% (34 contro 27), sul 5,6% (35 contro 28) e sul 5,8% (36 contro 32) delle battaglie del test. I seed 8, 9, 10 avevano già confermato Spikes + mosse su LightGBM: per 35 e 36 la conferma non è indipendente dalla scelta.

---

## Idee da provare (30/10/2026)

Idee non ancora provate, in tre filoni (feature nuove, togliere o trasformare feature, pipeline e modelli), unite dove
si sovrapponevano e ordinate per priorità. Da nessuna, da sola, ci aspettiamo un guadagno ≥ +0,5 punti.

| Priorità | Proposta |
|---|---|
| 1 | **P4 - Snapshot a turni precedenti** (provata nell'exp 09: negativa): feature calcolate anche sulla timeline tagliata a t = 20/24/27 come righe di train in più (stessa etichetta, `sample_weight` minore, colonna `snapshot_turn`); `StratifiedGroupKFold` per `battle_id`, valutazione solo sulle righe a t = 30. Richiede `N_TURNS` come parametro |
| 2 | **P1 - Qualità delle decisioni del giocatore** (provata nell'exp 09: negativa): efficacia di tipo media delle mosse d'attacco contro il Pokémon avversario in campo, % superefficaci / resistite, mosse di stato sprecate (bersaglio già con status o immune), qualità degli switch (il Pokémon entrato resiste alla mossa avversaria di quel turno) |
| 3 | **P6 - Vincoli monotoni + regolarizzazione + media su seed**: `monotone_constraints` su ~10-15 feature con direzione ovvia (`diff_hp_total`, `lost_est`, trend HP), regolarizzazione ritarata sulla CV ripetuta, 5 seed per membro dell'ensemble |
| 4 | **P3 - Pressione residua al turno 30**: Spikes ancora attive (non tolte con Rapid Spin) × Pokémon non immuni in panchina, contatore del Toxic, Leech Seed attivo, turni di sonno e sonno da Rest vs inflitto, Leftovers di P1 ancora in gioco |
| 5 | **P7 - LogReg additiva con spline** su ~20 feature robuste + poche interazioni, come membro diverso del soft voting |
| 6 | **P5 - Scambio P1/P2**: righe specchiate (etichetta 1 - y) solo nei fold di train, media delle due predizioni sul test. Giudicata la più rischiosa: P1 e P2 non hanno le stesse informazioni, le righe specchiate sarebbero fuori distribuzione |
| 7 | **P2 - Rendimento effettivo per Pokémon**: HP inflitti / subiti per turno in campo, aggregati sui Pokémon ancora in gioco. Rumorosa e ridondante con `exp_dmg`, trend HP e `team_matchup` |

Scartate: pseudo-labelling sul test (vietato), adversarial validation per scegliere feature (userebbe il
test), soglia di decisione (modello già calibrato), nuovo meta-learner per lo stacking (già provato nell'exp 04), k-NN
sulle traiettorie (k-NN fermo a 0,659), encoding a rapporti invece delle differenze (gli alberi li costruiscono già).

Validazione per tutte: CV ripetuta appaiata (seed 42, 7, 123), adozione con guadagno ≥ +0,5 punti e almeno 11 fold
su 15 migliorati, più lo stress test sui KO di P2.

---

## Idee da provare (31/10/2026, obiettivo 0,70 in leaderboard)

Tre filoni: segnale non sfruttato nei dati grezzi (con piccoli controlli descrittivi), controllo della pipeline, modelli,
ensemble e strategia di submission. Priorità: Q1 > Q3 > Q2 > Q4 > Q5.

**Controllo della pipeline: nessun bug.** HP totale ricostruito dalla cumulata = quello degli stati (differenza
1,8e-15); ogni Pokémon va KO una sola volta; parità train/test nella submission prudente; scostamento massimo delle
feature fra train e test 0,058 deviazioni standard. Due piccole perdite: ~45% degli switch volontari risultano
"move" perché nello stesso turno è registrata la mossa del Pokémon entrato (correlazione col residuo 0,015); `lost_est`
non vede i Pokémon di P2 spariti con HP > 0,5 (una correzione senza target non cambia nulla). Da citare nel report.

| Priorità | Proposta | Evidenza | Guadagno atteso |
|---|---|---|---|
| 1 | **Q1 - Forza della squadra (target encoding fuori fold)**: `te_p1` = tasso di vittoria smussato (K = 10) della squadra esatta di P1 (insieme delle 6 specie); `te_p2` = tasso delle squadre del train che contengono tutte le specie rivelate di P2; `diff_te`. Stimato dentro ogni fold, con cross-fitting interno sulle righe di train | Solo 4.167 squadre diverse in 10.000 battaglie; il 71% delle squadre di P2 rivelate per intero coincide con una squadra di P1 del train; dispersione dei tassi di vittoria oltre il rumore binomiale (0,098 vs 0,069); correlazione parziale 0,112 dato tutte le 155 feature (tutti i candidati precedenti ≤ 0,04) | +0,5 / +1,0 punti |
| 2 | Q3 - Media di modelli su più viste delle feature (v1, robuste, v2, v3, decisioni), ognuno con 5 seed | Le viste hanno un po' di segnale indipendente (residuo 0,03-0,04) | +0,1 / +0,3 |
| 3 | Q2 - Ricostruzione esatta dei Pokémon nascosti di P2 quando le specie rivelate individuano una sola squadra (13-20% delle battaglie) | Exp 07 neutro | +0,1 / +0,2 |
| 4 | Q4 - Membri strutturalmente diversi e più regolarizzati (LGBM extra_trees, CatBoost Lossguide, SVM-RBF sulle robuste) | - | 0 / +0,2 |
| 5 | Q5 - Conteggio totale degli switch per lato | Residuo 0,015 | ≤ +0,1 |

**Controlli richiesti per Q1:** encoding stimato solo sul train di ogni fold, con
cross-fitting interno; CV ripetuta appaiata con encoding ricalcolato per ogni seed; CV raggruppata per squadra di P1
(le squadre di validazione non compaiono nel train); etichette o chiavi delle squadre mescolate (il guadagno deve
sparire); encoding forzato a 0,5 su 30/60/100% delle righe di validazione (squadre mai viste, come potrebbe
succedere sul set privato). Rischio principale: il set privato potrebbe non riusare lo stesso insieme di squadre.

**Processo:** fissare `REPO_REF` al commit che ha prodotto ogni submission finale; scegliere le
3 submission finali per CV e robustezza, usando la leaderboard solo per scovare bug.

---

## Idee da provare (01/11/2026)

Tre filoni: identità di squadre e specie, controllo del codice recente, aspetti mai considerati. R1 e R3 hanno la
priorità più alta, quasi alla pari.

| Priorità | Proposta | Evidenza (diagnostica su LogReg, non LightGBM) |
|---|---|---|
| 1 | **R1 - Encoding per coppie di specie**: ogni squadra del pool contribuisce a tutte le sue coppie di specie (y - 0,5 da P1, 0,5 - y da P2 rivelato per intero); `pr1` e `pr2` = media sulle coppie della somma / (n + 20) | Con le squadre di validazione tolte dal pool: encoder -0,15, encoder + coppie +0,28 (+0,42 sull'encoder, 11/15). Su CV normale +0,21 sull'encoder (9/15) |
| 2 | **R3 - Mascheramento per lato**: `train_unseen` estratto separatamente per P1 e P2, stress test solo su P1, `n_inner=20` | Dopo il mascheramento le righe con n1 = 0 e n2 > 0 vincono il 36,4%: la rarità sopravvive. KS dei conteggi train/validazione da 0,062 a 0,029 |
| 3 | R6 - Nessun altro lavoro sulle feature, si passa a misurare le configurazioni finali e alle consegne | - |
| 4 | R2 - Squadre di P2 rivelate in parte nel pool, distribuite sulle squadre compatibili | corr(`diff_te`, y) fuori fold da 0,137 a 0,157 |
| 5 | R4 - Conteggio dei KO non registrati di P2 (cambio di Pokémon, il vecchio non torna, il nuovo ha HP pieni, P1 ha attaccato) | +0,07 ± 0,40 (8/15) |
| 6 | R5 - Togliere dal train 17 coppie di battaglie duplicate e 2 specchiate | Atteso sotto 0,1 punti |

Decisione: provare **R1 + R3 in un solo esperimento**, una sola esecuzione senza iterare, e poi passare a R6
comunque vada.

**Controllo del codice.** Nessun leakage: cambiando le etichette di un fold interno le codifiche di quel fold non
cambiano. Punti deboli confermati: il pool è sbilanciato (P2 entra solo se rivelato per intero, e allora P1 vince il
55,1%); lo stress test maschera entrambi i lati insieme, quindi non rappresenta un set privato; il controllo con chiavi
mescolate non può rilevare un leak che passa dall'identità della squadra (duplicati e rivincite verificati a parte:
effetto trascurabile). L'exp 09 è stato un test equo. La stima corretta di LightGBM + squadra è 0,6867 sui 15 fold, non
0,6905 (solo seed 42).

**Risultati negativi, per il report.** Ricostruzione esatta dei Pokémon nascosti di P2, matchup squadra contro
specie, livello di singola specie, vicini a una specie di distanza, modelli separati per stile di gioco, ordine delle
righe, bot con strategia fissa (i prefissi di 5 turni sono quasi tutti unici), due formati di HP: nessun guadagno.

**Asimmetria sui KO.** Il 98,2% delle righe con `fnt` di P2 ha anche `fnt` di P1, quindi sembra un artefatto di
registrazione più che di stile di gioco; il meccanismo esatto è una deduzione.

**Scartate perché vietate o scorrette.** Contare le squadre sul set da predire (usa il test: ammesso solo se lo
autorizzano i docenti); copiare le etichette delle 8 battaglie del test identiche a battaglie del train e delle 4
specchiate.

---

## Varianti dell'exp 14 (02/11/2026)

Tre aspetti controllati: codice e leakage, guadagno atteso e aderenza della CV, strategia. Fatto nuovo: la classifica finale usa
l'altro 70% del test. Priorità delle opzioni:

| Priorità | Opzione |
|---|---|
| 1 | Media di modelli con encoder diversi |
| 2 (pari) | Variante C `residual` |
| 2 (pari) | Variante B `soft_p2` |
| 2 (pari) | A + B |
| 5 | Fermare il lavoro sulle feature |
| 6 | A + B + C |
| 7 | Variante A `n_inner=20` |

**R1 senza maschera** (`train_unseen=0`) è stata aggiunta dopo come candidato. Nell'exp 11 la maschera
cambiava poco sulla CV normale (+0,90 senza, +0,88 con `train_unseen=0.3`), quindi il guadagno atteso è piccolo.

**Misure (diagnostica su LogReg, non LightGBM; dentro il rumore).**
- Squadre di P1 mai viste: test 26,9%, leave-one-out del train 27,3%, validazione in CV 29,2%; righe di train col conteggio a 0 con la maschera
  attuale circa 52%. La CV riproduce bene il test ed è al massimo circa 0,1 punti pessimista.
- Guadagno su R1 + R3, LogReg. Prima misura (seed 42): A -0,02, B -0,01, C +0,21, A + B +0,13, A + B + C +0,10. Seconda misura: A +0,13, B +0,11, C +0,17,
  A + B +0,04, A + B + C +0,19. I segni non concordano: nessuna variante ha un'evidenza solida.
- `n_inner=20` raddoppia l'anticorrelazione dentro la squadra (`corr(te_p1, y)` da -0,27 a -0,55 nelle squadre con 20-49 battaglie).

**Problemi nel notebook 30, da correggere prima dell'esecuzione.** Il controllo "own label" guarda solo `pr1`/`pr2`, che B non tocca; le chiavi
mescolate non vedono un leak di train e girano solo sul candidato migliore; nello scenario "unseen both" le chiavi parziali di validazione
restano nel pool e favoriscono B; il testo "most battles teach nothing" è falso, perché il 55,6% delle battaglie entra già tramite P2.

**Regola.** La condizione sulle squadre mai viste diventa solo descrittiva. Si tiene +0,3, con conferma del vincitore su seed nuovi.

**Altre proposte.** LogReg + encoder come membro della media; media su seed di encoder e modello, mai provata da sola. Verificare ora che
il notebook finale rigeneri i CSV (`REPO_REF` non è fissato, le librerie su Kaggle possono differire). Finali proposte: 24, 28 e la media se
confermata, altrimenti 23; la 06 non serve più come copertura.

Stima, non misura: risultato finale atteso circa 0,69-0,696; 0,71 improbabile (circa 1-5%).

---

## Idee da provare (02/11/2026, obiettivo 0,71)

Tre filoni: segnale nella timeline grezza, struttura del dataset, impostazione dei modelli e limite raggiungibile. Ogni idea
è stata misurata subito. Tutto è diagnostica su modelli veloci (LogReg, Random Forest, HistGB), **nulla è misurato su LightGBM**, e i fold ricampionano le stesse 10.000 battaglie.

| Proposta | Misura (contro il riferimento con encoder R1 + R3, salvo dove indicato) | Guadagno atteso su LightGBM (stima) |
|---|---|---|
| **Spikes attive al turno 30** (`p1_on`, `p2_on`, `diff_on`): Spikes accende il lato avversario, Rapid Spin spegne il proprio, in ordine di turno. Nessun fit sul target | LogReg +0,65 (15/15), seed 1/2/3 +0,49 (13/15); Random Forest +0,42 (12/15); colonna permutata +0,01; soli conteggi senza ordine +0,33 | +0,3/+0,6 |
| **Righe specchiate nel train** (scambio P1/P2, y → 1 - y, media con la predizione specchiata; era P5 della prima lista, mai misurata) | HistGB (riferimento senza specchio) +0,35 (25/30); KO di P2 alterati +0,42; `te` neutralizzate +0,22 | +0,1/+0,3 |
| **Tipicità delle mosse** (`lp`): per lato, media di log p(mossa, specie), dentro i fold e senza la riga propria | LogReg +0,41 (11/15), migliore fra 7 varianti; con encoding di (specie, mossa) +0,44 (13/15), seed 1/2/3 +0,47; frequenze mescolate -0,05 | +0,1/+0,4 |
| Mosse firmate (n-grammi), `p1_alive_leftovers`, Thief, Heal Bell | Su `spikes_on`: +0,07-0,16; +0,30, ma Random Forest -0,01 | 0/+0,1 |
| Flag di fine battaglia (P2 `fnt` da solo al turno 30; 6 KO di P1) | +0,06 (7 migliorati, 4 peggiorati) | ≤ +0,1 |

**Possibile sovrapposizione:** la conformità del moveset su `spikes_on` aggiunge solo +0,08 e +0,28 e gli n-grammi erano in gran parte un surrogato delle Spikes, quindi i guadagni di
Spikes e `lp` probabilmente non si sommano.

**Spiegazioni, dedotte e non provate.** Le squadre uniche sembrano giocatori occasionali: rarità simmetrica (P1 unica contro P2 comune vince il 41%, P1 comune contro P2 mai vista il 69%), mosse fuori
standard. L'asimmetria sui KO potrebbe nascere dal fatto che la timeline è fotografata alle decisioni di P1.

**Da non usare senza il parere dei docenti.** Lo slot di P1 segue l'ordine di comparsa dei Pokémon (monotono nel 94,7% dei casi), quindi l'ordine degli slot rivela informazione successiva al turno 30, in circa l'11% delle battaglie. Non è stato misurato.

**Non ha funzionato.** Effetti al turno 30, schermi, meteo, sonno/Toxic in dettaglio, switch reali, serie di HP, ruoli vivi, danno realizzato, oggetti (assorbiti dall'encoder); Bradley-Terry sulle squadre, testa a testa, KO di P2 dedotti, slot saltati, tipicità delle coppie di specie; rating logistico, segmenti, interazioni HP × squadra, GAM, proiezione dal turno 21, probabilità lungo la battaglia, perdite e pesi diversi, soglia.

**Limite raggiungibile (stime).** Learning curve: circa 0,690-0,692 con 10.000 battaglie. Livello vero attuale stimato circa 0,694 ± 0,005; P(finale ≥ 0,71) circa 4%, P(finale ≥ 0,70) circa 26%. Il primo in classifica: gran parte del vantaggio può essere selezione fra molte squadre (simulazione con ipotesi sulla dispersione delle squadre).
