# Soft actor critic (SAC)
## Descrizione dell' algoritmo
Soft actor critic è un algoritmo di deep reinforcement learning off-policy derivato da *DQN (Deep Q Learning)* e con esso condivide le seguenti funzionalità:
* La Q function viene approssimata da una rete neurale *(input: (stato, azione), output: valore)*
* Non si utilizza un modello di transizione, ma si acquisisce conoscenza dall' esplorazione
* L' agente conosce lo stato in cui si trova (o comunque una descrizione parziale dello stato attuale)
* L' agente conosce le azioni che è in grado di compiere

SAC però inserisce il concetto di entropia all' interno del funzionamento dell' algoritmo.
L' entropia definisce la *casualità* della policy: più è elevata, più l' output della policy diventa imprevedibile, viceversa un entropia pari a 0 significherebbe che la policy è deterministica (prevedibilità massima).
L' entropia della policy nello stato s è formalmente definita da **H(π(.|s)) = ∑P(π(a|s))\*(-log(P(π(a|s))) per ogni a possibile da parte dell' agente**, ovviamente per questo motivo l' algoritmo è progettato per lavorare con policy stocastiche (per lo meno durante la fase di training), altrimenti questo termine sarebbe sempre par a 0.

SAC presenta le seguenti peculiarità:
* Questa implementazione di SAC funziona solamente per uno spazio di azioni continuo
* La policy è implementata tramite una rete neurale con un determinato set di pesi *(input: stato, output: azione)*
    * Per questi 2 motivi la policy non viene aggiornata selezionando l' azione che massimizza la Q function per un determinato stato, ma aggiornando i pesi della policy seguendo il gradiente calcolato su V(s) rispetto ai pesi della policy
    *  V(s) viene approssimato usando la Q function, la quale in input ha lo stato attuale e l' azione viene scelta usando il *reparametrization trick* che verrà discusso in seguito
* Utilizza 2 Q function *Q1 e Q2*, ciascuna delle quali ha la corrispondente target network (come per l' algoritmo *TW3*)
	* Ogni aggiornamento coinvolge entrambe le Q, ma per l' aggiornamento della policy e per il calcolo dei target si usa sempre il valore minore tra le due per la stessa coppia stato-azione (per evitare picchi anomali possibili nella funzione Q)
* Oltre al reward, SAC include l' entropia per il calcolo della *value function* (e di conseguenza la *Q function*)
* Il Reward cumulato per V viene riscritto come **R(s,a,s') + α\*H(π(a|s))** dove α sarebbe il coefficiente di tradeoff: un valore che permette di dare più o meno priorità all' entropia rispetto al reward.
* Nell' implementazione però vengono considerate 2 approssimazioni:
	* Q(s,a) non considera l' entropia per il reward istantaneo (quindi viene considerato solo r)
	* Q(s,a) approssima l' entropia **H(π(.|s))** calcolandola come **-α\*log(π(a'|s))**
	* Di conseguenza **Q(s,a) = r + γ\*Q(s',a')-α\*log(π(a'|s'))) dove a' è un' azione etratta dalla policy stocastica π nello stato s'**
* La value function viene approssimata con **V(s)=tanh(μ(s)+normal(mean=0, std=1))** dove μ(s) è la policy deterministica calcolata dalla rete neurale (questa approssimazione è chiamata *reparametrization trick*)

L' aggiornamento delle Q, Q target e policy avviene dopo ogni azione da parte dell' agente (dopo le prime 100) tramite batch normalization e polyak update (quest' ultimo però riguarda solo Q target e in pratica sarebbe un aggiornamento pesato della Q target tramite gradient descent dove un peso 𝜏 compreso tra 0 e 1, viene usato con la formula *Qtarget = 𝜏 \* Q + (1-𝜏) Qtarget*)
## Descrizione dell' environment
L' environment su cui si è sperimentato l' algoritmo è il LunarLanderContinuous-v2 caratterizzato da:
* agente: lunar lander simulato
* action space: 2 attributi float (lista):
	0. main engine: indica la potenza del motore che o è spento o applica un' accelerazione al lunar lander verso l' alto (valori compresi fra 0.5 e 1 quando è acceso indicano la potenza, sotto 0.5 il motore è spento)
	1. side engines: quando è compreso fra -1 e -0.5 applica un accelerazione da parte dell' engine sinistro (verso destra) e quando è compreso tra 0.5 e 1 l' accelerazione è applicata dall' engine destro (verso sinistra), mentre i valori compresi tra -0.5 e 0.5 indicano che l' engine è spento
* stato: 6 attirbuti float (lista)
	0. coordinate orizzontali (0 nello stato iniziale)
	1. coordinate verticali (0 nello stato iniziale)
	2. velocità orizzionale
	3. velocità verticale
	4. angolo di rotazione
	5. velocità angolare
	6. 1.0 se la prima gamba del lunar lander è appoggiata a terra, 0.0 altrimenti
	7. 1.0 se la seconda gamba del lunar lander è appoggiata a terra, 0.0 altrimenti
* rewards: 
	* -0.3 per ogni frame nel quale l' engine usa il main engine
	* -0.03 per ogni frame nel quale l' engine usa uno dei side engines
	* -100 lander incidentato (fine episodio)
	* +10 per ogni gamba che appoggia sul terreno
	* +100 per l' atterraggio

## Obiettivo (trattazione informale)
SAC grazie all' entropia permette di incentivare le azioni poco probabili per la policy stocastica attuale, che ovviamente restituisce azioni che non rispecchiano una conoscenza completa dell' ambiente.
Questo permette una maggiore esplorazione, così il training può coinvolgere più coppie stato-azione e la policy imparata dovrebbe riflettere il comportamento migliore nelle situazioni non esplorate, perchè più generalizzata (*policy smoothing*).
Lo scopo del trade-off coefficient è quello di dare la possibilità di diminuire il peso dell' entropia sul valore degli stati e dare più priorità all' exploitation.

## Tweaking dei parametri:
Per il training ci sono diversi tipi di parametri modificabili in questa sezione li elencherò con una breve descrizione
* numero dei episodi per training: ho notato che il trend dei reward accumulati non subisce grosse variazioni dopo circa l' episodio 500 per ogni test, quindi ho preferito non andare oltre i 1001 episodi (anche per evitare overfitting)
* std_scale (scaling progressivo della std della policy): booleano che indica se il valore della rumore associato alla policy stocastica diminuirà (True) o rimarrà costante (False) durante il training.
* std_scaling_type: tipo di scaling usato per il rumore durante il training (verrà discusso in modo approfondito nella trattazione dei test)
* std_decay: valore decimale < 1 che indica un fattore da moltiplicare per il rumore dopo ogni episodio (se il tipo di scaling è standard).
* std_min: valore minimo di rumore.
* std: valore iniziale della std.
* Il coefficiente di trade-off alpha presenta gli stessi parametri del rumore std (valore iniziale, scaling si/no, valore minimo, tipo di scaling, fattore di scaling)
* buffer size: massimo numero di elementi presenti nel replay buffer
* batch: numero di tuple (s, a, r, s', done) usate per il calcolo delle loss functions (experience replay)
* actor: parametri della rete neurale che rappresenta la policy:
	* numero di hidden layers (lasciato sempre a 2)
	* nodi per ogni hideen layer
* critic: parametri per la rete neurale che rappresenta le funzioni Q
	* numero di hidden layers collegati solo allo stato di input (o azione di input)
	* numero di nodi per per hidden layer collegati solo allo stato di input (o azione di input)
	* numero di hidden layers collegati sia a stato che azioni
	* numero di nodi per ogni hidden layer generico

## Discussione dei test