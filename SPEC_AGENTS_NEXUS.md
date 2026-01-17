# Spécification Technique : Intégration d'Agents Dynamiques dans Nexus-Dev

## 1. Vue d'ensemble
Ce document décrit l'architecture pour transformer `nexus-dev` (serveur MCP de contexte) en un **orchestrateur d'agents autonomes**.
L'objectif est de permettre l'ajout d'agents spécialisés simplement en déposant un fichier de configuration YAML dans un dossier, sans modifier le code source du serveur.

### Flux de données
1.  **Chargement :** Au démarrage, le serveur scanne `./agents/*.yaml`.
2.  **Exposition :** Chaque agent est converti dynamiquement en un **Outil MCP** (ex: `ask_agent_reviewer`).
3.  **Exécution :**
    * L'agent reçoit une tâche.
    * Il interroge la mémoire `Nexus` (LanceDB) pour obtenir du contexte (RAG).
    * Il construit un prompt système structuré.
    * Il utilise un LLM (via LiteLLM) pour exécuter la tâche et appeler des outils si nécessaire.

---

## 2. Structure des Fichiers
Nouveaux fichiers à créer ou modifier :
- `models/agent_config.py` : Définitions Pydantic.
- `core/prompt_factory.py` : Génération des prompts avec balises XML.
- `core/agent_executor.py` : Logique d'exécution (LiteLLM + RAG).
- `core/agent_manager.py` : Chargement et gestion du cycle de vie.
- `server.py` : Modification pour l'enregistrement dynamique des outils MCP.

---

## 3. Détails d'Implémentation

### A. Modèles de Configuration (`models/agent_config.py`)
Utiliser `pydantic` pour une validation stricte des YAML.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class AgentProfile(BaseModel):
    role: str
    goal: str
    backstory: str
    tone: str = "Professionnel et direct"

class AgentMemory(BaseModel):
    enabled: bool = True
    rag_limit: int = 5
    topics: List[str] = Field(default_factory=list) # Filtres pour LanceDB

class LLMConfig(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 0.5
    max_tokens: Optional[int] = 4000

class AgentConfig(BaseModel):
    name: str           # ID interne (ex: "CodeReviewer")
    display_name: str   # Pour l'UI
    description: str    # Description pour le routeur d'outils MCP
    profile: AgentProfile
    memory: AgentMemory
    tools: List[str]    # Liste des noms d'outils autorisés
    llm_config: LLMConfig
```

### B. Moteur de Prompt (`core/prompt_factory.py`)

Le prompt doit utiliser une structure XML stricte pour séparer le contexte des instructions.

```python
from typing import List
from models.agent_config import AgentConfig

class PromptFactory:
    @staticmethod
    def build(agent: AgentConfig, context_items: List[str]) -> str:
        # Formattage du contexte RAG
        memory_block = ""
        if context_items:
            items_str = "\n".join([f"- {item}" for item in context_items])
            memory_block = f"""
<nexus_memory>
INFORMATIONS CONTEXTUELLES DU PROJET (RAG) :
{items_str}
Utilise ces informations pour guider tes actions.
</nexus_memory>
"""

        return f"""
<role_definition>
Tu es {agent.display_name}.
ROLE: {agent.profile.role}
OBJECTIF: {agent.profile.goal}
TON: {agent.profile.tone}
</role_definition>

<backstory>
{agent.profile.backstory}
</backstory>

{memory_block}

<instructions>
1. Analyse la demande de l'utilisateur.
2. Utilise les outils disponibles si nécessaire.
3. Si tu génères du code, respecte les contraintes définies dans <nexus_memory>.
</instructions>
"""
```

### C. Exécuteur (`core/agent_executor.py`)

Le moteur qui orchestre LiteLLM et Nexus.

**Responsabilités :**

1. **Retrieve :** Chercher dans LanceDB les "leçons" pertinentes basées sur la requête utilisateur.
2. **Augment :** Créer le prompt via `PromptFactory`.
3. **Generate :** Appeler `litellm.completion` avec `tools=...`.

```python
import litellm
from models.agent_config import AgentConfig
from core.prompt_factory import PromptFactory

class NexusAgentExecutor:
    def __init__(self, config: AgentConfig, db_session):
        self.config = config
        self.db = db_session

    async def execute(self, user_task: str):
        # 1. RAG Retrieval (Simulation ou appel réel à Nexus)
        # context = await self.db.search(user_task, limit=self.config.memory.rag_limit)
        context = ["Exemple de contexte: Utiliser Python 3.11"] # Placeholder

        # 2. Construction Prompt
        system_prompt = PromptFactory.build(self.config, context)

        # 3. Appel LLM (avec gestion des outils à implémenter)
        response = await litellm.acompletion(
            model=self.config.llm_config.model,
            parameters={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_task}
                ],
                "temperature": self.config.llm_config.temperature
            }
        )
        
        return response.choices[0].message.content
```

### D. Gestionnaire (`core/agent_manager.py`)

Charge les YAML.

```python
import yaml
from pathlib import Path
from models.agent_config import AgentConfig

class AgentManager:
    def __init__(self, agents_dir: str = "./agents"):
        self.agents_dir = Path(agents_dir)
        self.agents = {}
        self.load_agents()

    def load_agents(self):
        if not self.agents_dir.exists():
            return
        
        for f in self.agents_dir.glob("*.yaml"):
            with open(f) as yf:
                data = yaml.safe_load(yf)
                agent = AgentConfig(**data)
                self.agents[agent.name] = agent
```

### E. Intégration Serveur (`server.py`)

Enregistrement dynamique des outils au démarrage.

```python
# Dans la configuration du serveur MCP FastMCP ou similaire

agent_manager = AgentManager()

# Boucle d'enregistrement dynamique
for agent_name, agent_cfg in agent_manager.agents.items():
    
    # On crée une closure pour capturer la config de l'agent
    def create_tool_function(cfg):
        async def agent_wrapper(task: str) -> str:
            """Exécute une tâche via l'agent configuré."""
            executor = NexusAgentExecutor(cfg, db_session) # db_session doit être accessible
            return await executor.execute(task)
        return agent_wrapper

    # Enregistrement via le décorateur ou la méthode .tool() do serveur MCP
    # Note: Le nom de l'outil doit être unique et sans espaces
    tool_name = f"ask_{agent_name.lower()}"
    server.tool(name=tool_name, description=agent_cfg.description)(create_tool_function(agent_cfg))
```

---

## 4. Exemple de YAML (`agents/doc_expert.yaml`)

```yaml
name: "DocuBot"
display_name: "Expert Documentation"
description: "Déléguer la rédaction ou la correction de documentation technique."

profile:
  role: "Technical Writer Lead"
  goal: "Assurer une documentation claire et à jour."
  backstory: "Expert rigoureux, tu privilégies le Markdown et les docstrings Google Style."
  tone: "Pédagogique"

memory:
  enabled: true
  rag_limit: 3
  topics: ["documentation", "conventions"]

tools:
  - "read_file"
  - "write_file"

llm_config:
  model: "claude-3-5-sonnet"
  temperature: 0.2
```

## 5. Instructions pour l'Assistant IA

1. Créer d'abord les modèles Pydantic pour valider la structure.
2. Implémente le chargement YAML.
3. Intègre `litellm` pour l'exécution.
4. Modifie `server.py` pour boucler sur les agents chargés et créer les outils MCP correspondants.
5. Assure-toi que les exceptions sont gérées (ex: fichier YAML malformé).
