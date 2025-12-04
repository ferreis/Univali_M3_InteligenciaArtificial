**Projeto: Previsão de Batalhas RPG com RNN**

Este repositório contém uma implementação em C de uma Rede Neural Recorrente (RNN) simples para prever a probabilidade de vitória de um herói em uma simulação curta de batalha RPG. O arquivo principal do projeto é `main.c`.

**Descrição do Problema:**
- **Contexto:** Prever o vencedor de um confronto RPG com base nos atributos dos personagens (HP, Mana, Força, Inteligência, Defesa, Defesa Mágica) ao longo dos últimos 3 turnos.
- **Justificativa:** Problemas sequenciais (como decisões em batalhas turn-based) beneficiam-se de modelos com memória temporal. Uma RNN básica permite capturar dependências entre turnos sem exigir frameworks externos, sendo adequada para aprendizado didático e experimentação local.

**Base de Dados Escolhida:**
- **Origem:** Os dados são gerados por simulação no próprio arquivo `main.c` (função `simular_gameplay`). Não há uma base externa — o autor constrói dois cenários extremos (vitória fácil e derrota certa) e utiliza as sequências geradas como exemplos de treinamento.
- **Formato:** Cada amostra é uma sequência de 3 turnos (constante `SEQ_LEN = 3`). Cada turno contém 12 entradas (`NUM_ENTRADAS = 12`): 6 atributos do herói e 6 atributos do monstro, normalizados dividindo por 100.

**Pré-processamento:**
- **Normalização:** Os atributos são normalizados dividindo por 100 dentro de `simular_gameplay`, reduzindo escala e ajudando a função de ativação.
- **Sequência fixa:** O modelo espera sequências de comprimento 3; inputs são organizados como `entrada[SEQ_LEN][NUM_ENTRADAS]`.
- **Sem split explícito:** O código atual usa os dois cenários gerados repetidamente para treinar (não há divisão formal em treino/validação/teste). Para experimentos reproducíveis é recomendado gerar um conjunto maior e aplicar splits.

**Modelagem (Técnica de IA escolhida):**
- **Tipo de rede:** Rede Neural Recorrente (RNN) simples implementada manualmente em C.
- **Arquitetura:**
  - Camada de entrada: `NUM_ENTRADAS = 12` entradas por passo de tempo.
  - Camada oculta recorrente: `NUM_OCULTOS = 8` neurônios.
  - Saída: `NUM_SAIDAS = 1` (probabilidade de vitória, ativação sigmoide).
  - Funções de ativação: tangente hiperbólica (`tanh`) para estados ocultos, sigmoide para saída.
  - Mecanismo de memória: pesos recorrentes `Wh[NUM_OCULTOS][NUM_OCULTOS]` e estados ocultos `estados_ocultos[SEQ_LEN+1][NUM_OCULTOS]`.
- **Treinamento:** Backpropagation Through Time (BPTT) implementado manualmente na função `treinar`. Parâmetros importantes:
  - `EPOCAS = 20000`
  - `TAXA_APRENDIZADO = 0.1`
  - Inicialização aleatória dos pesos (função `aleatorio`).

**Resultados obtidos (simulações):**
- **O que o código faz:** Gera dois cenários com `simular_gameplay`, treina a RNN alternando entre exemplo de vitória (`alvo = 1.0`) e exemplo de derrota (`alvo = 0.0`) por `EPOCAS` iterações e, ao final, permite executar simulações interativas onde o usuário entra com atributos e a RNN prevê a probabilidade de vitória.
- **Observações sobre resultados:**
  - Com apenas dois exemplos (um de vitória e um de derrota) a rede aprende um comportamento muito limitado e probabilístico: ela tende a ajustar seus pesos para separar esses casos extremos, mas não generaliza bem para casos intermediários.
  - Em execuções, a função `forward` retorna um valor entre 0 e 1 indicando a probabilidade estimada de vitória; o programa imprime um diagnóstico simples (>0.7 vantagem, <0.3 desvantagem, caso contrário empate).

**Como compilar e executar:**
```bash
# Compilar
gcc main.c -o rpg -lm

# Executar
./rpg
```

**Limitações conhecidas:**
- Dataset extremamente pequeno (apenas 2 exemplos), o que impede avaliação estatística e generalização.
- Nenhuma validação/curva de aprendizado é registrada nem métricas (acurácia, loss) são salvas.
- Treinamento em C é educativo, mas dificulta experimentação rápida; frameworks (Python + TensorFlow/PyTorch) permitem análise mais completa.

**Melhorias e próximos passos (sugestões):**
- **Gerar dataset maior:** Automatizar geração de centenas/ milhares de batalhas com variação aleatória nos atributos, e salvar amostras (entrada/label) para treino e validação.
- **Dividir dataset:** Criar splits treino/val/test e acompanhar métricas como loss e acurácia.
- **Salvar pesos:** Implementar serialização dos pesos da RNN para evitar re-treinar do zero a cada execução.
- **Avaliação:** Calcular ROC/AUC e matriz de confusão (convertendo problema em classificação binária com threshold apropriado).
- **Experimentar arquiteturas:** LSTM/GRU para capturar dependências mais longas; aumentar `SEQ_LEN` para olhar mais turnos.
- **Portar para Python:** Reimplementar em Python usando PyTorch/TensorFlow para acelerar experimentação e usar ferramentas de visualização (TensorBoard, matplotlib).

**Referência de Arquivo:**
- O código principal e a lógica estão em `main.c` (ver funções: `simular_gameplay`, `forward`, `treinar`, `inicializar_rnn`).

Se desejar, eu posso:
- Gerar um dataset sintético maior e adaptar `main.c` para carregar amostras a partir de um arquivo CSV.
- Reescrever o modelo em Python (PyTorch) para facilitar experimentação e avaliação.

Arquivo criado: `Read.md`.
