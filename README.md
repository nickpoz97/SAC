# Soft actor critic (SAC)
## Descrizione dell' algoritmo
Soft actor critic è un algoritmo di deep reinforcement learning off-policy derivato da *DQN (Deep Q Learning)* e con esso condivide le seguenti funzionalità:
* La Q function viene approssimata da una rete neurale *(input: (stato, azione), output: valore)*
* Non si utilizza un modello di transizione, ma si acquisisce conoscenza dall' esplorazione
* L' agente conosce lo stato in cui si trova (o comunque una descrizione parziale dello stato attuale)
* L' agente conosce le azioni che è in grado di compiere

SAC però differisce da DQN per i seguenti aspetti:
* Questa implementazione di SAC funziona solamente per uno spazio di azioni continuo
    * Per questo motivo la policy non viene aggiornata 
* Utilizza 2 target network Q1 e Q2, scegliendo per ogni target quella con il valore più basso a parità di stato e azione (per evitare di considerare pèicchi anomali)