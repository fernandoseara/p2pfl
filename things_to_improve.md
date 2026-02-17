- Responsabilidades de workflow/experiment/status
    - por ejemplo ROUND se accede usando el workflow y no se si es lo mas adecuado
    - la forma de loguearlo tiene que ser a modo observer, mucho mas directo desde un estado centralizado - ESTO ES MUUUUY IMPORTANTE
- Mejorar contratos ideales a la hora de definir los workflows




- Almacenar Estado
    - CADA UNO QUE HAGA LO QUE QUIERA
        - Cuidado CON LOGGING
    - TODA LA INFO RELEVANTE VAYA ESPERIMENT
    - LOGGER A WEB SEA UN OBSERVADOR
    - INCENTIVARLO - NO ES LO UNICO - PUEDES PICAR UN ATRIBUTO EN EL WORKFLOW Y YA (PERO NO SE LOGUEA)
    - BasicPeerState mergear

- Como se pasa la informacion relevante
    - ahora mismo todo NODE

    - OPTION 1
    CLASE BASE
    async def execute(node=, experiment=, commin=, **kwargs) -> list[tuple[str, float]]:

    CLASE ESPECÍFICA
    async def execute(nocommin=) -> list[tuple[str, float]]:

    - OPTION 2
    TODO EN EL WORKFLOW
    WORFLOW<-STAGE

- Simplificar WF