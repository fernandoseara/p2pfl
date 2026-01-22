- worflow.aggregete
- callbacks
- todo demasiado desperdigado, nombre no esta en el stage, hacen falta callbacks y otra clase que haga el workflow
- en el p2pfl model tenia un _wrapped, ns muy bien que era eso, simplemente apuntado por si da problemas
- trainset_size no esta en local state
- ahora epochs_per_round, ns si me gusta
- self.nei_status por ejemplo en network sattus no en node_state, con self.sending_models deberia ser igual
- self.status = "Idle" -> este estaus se deberia obtener del workflow
    - se tiene que revisar el workflow de actualizacion del estado
- delegaciones en el worflow del comando. Deja de tener tanta responsabilidad el comando como tal
- def __aggregate_votes borrado entero
    - deberia estar manejado en el workflow
- simulation: bool = False no longer needed
- commands = [] eliminado de node (factoria???)
- self.__stop_learning funciona???
- presend model sin hacer
- no se hasta que punto se usa
```
# Error tracking
self.failed: bool = False
self.error: Exception | None = None
```
