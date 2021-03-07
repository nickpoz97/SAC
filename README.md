# Soft actor critic (SAC)
## Descrizione dell' algoritmo
Soft actor critic è un algoritmo di deep reinforcement learning off-policy derivato da *DQN (Deep Q Learning)* e con esso condivide le seguenti funzionalità:
* La Q function viene approssimata da una rete neurale *(input: (stato, azione), output: valore)*
* Non si utilizza un modello di transizione, ma si acquisisce conoscenza dall' esplorazione
* L' agente conosce lo stato in cui si trova (o comunque una descrizione parziale dello stato attuale)
* L' agente conosce le azioni che è in grado di compiere

SAC però presenta le seguenti peculiarità:
* Questa implementazione di SAC funziona solamente per uno spazio di azioni continuo
* La policy è implementata tramite una rete neurale con un determinato set di pesi *(input: stato, output: azione)*
    * Per questi 2 motivi la policy non viene aggiornata selezionando l' azione che massimizza la Q function per un determinato stato, ma aggiornando i pesi della policy seguendo il gradiente calcolato su V(s) rispetto ai pesi della policy
    *  V(s) viene approssimato usando la Q function, la quale in input ha lo stato attuale e l' azione viene scelta usando il *reparametrization trick* che verrà discusso in seguito
* Utilizza 2 Q function *Q1 e Q2*, ciascuna delle quali ha la corrispondente target network (come per l' algoritmo *TW3*)
	* Ogni aggiornamento coinvolge entrambe le Q, ma per l' aggiornamento della policy e per il calcolo dei target si usa sempre il valore minore tra le due per la stessa coppia stato-azione
* Oltre al reward, SAC include l' entropia per il calcolo della *value function* (e di conseguenza la *Q function*)

L' entropia definisce la *casualità* della policy: più è elevata, più l' output della policy diventa imprevedibile, viceversa un entropia pari a 0 significherebbe che la policy è deterministica (prevedibilità massima).
L' entropia della policy nello stato s è formalmente definita da **H(π(.|s)) = ∑P(π(a|s))\*(-log(P(π(a|s))) per ogni a possibile da parte dell' agente**, ovviamente per questo motivo l' algoritmo è progettato per lavorare con policy stocastiche (per lo meno durante la fase di training), altrimenti questo termine sarebbe sempre par a 0.
* Il Reward cumulato per V viene riscritto come **R(s,a,s') + α\*H(π(a|s))**
* Nell' implementazione però vengono considerate 2 approssimazioni:
	* Q(s,a) non considera l' entropia per il reward istantaneo (quindi viene considerato solo r)
	* Q(s,a) approssima l' entropia **H(π(.|s))** calcolandola come **-α\*log(π(a'|s))**
	* Di conseguenza **Q(s,a) = r + γ\*Q(s',a')-α\*log(π(a'|s'))) dove a' è un' azione etratta dalla policy stocastica π nello stato s'**
* Infine la value function viene approssimata con **V(s)=tanh(μ(s)+normal(mean=0, std=1))** dove μ(s) è la policy deterministica calcolata dalla rete neurale (questa approssimazione è chiamata *reparametrization trick*)