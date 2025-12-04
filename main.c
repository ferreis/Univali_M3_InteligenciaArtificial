/*
 * MEU PROJETO DE IA: Previsão de Batalhas RPG com RNN
 * AUTOR: Rafael Fernando dos Reis Mecabô
 * OBJETIVO: Criei este código para simular lutas e usar uma IA 
 * que aprende com o tempo (RNN) para prever o vencedor.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- MINHAS CONFIGURAÇÕES ---
// Decidi olhar os últimos 3 turnos para a IA entender o "ritmo" da luta
#define SEQ_LEN 3
// São 12 entradas porque passo todos os atributos (HP, Mana, Força, Defesa...) do Heroi e do Monstro
#define NUM_ENTRADAS 12
// Escolhi 8 neurônios na camada oculta para a rede ter capacidade de processar esses dados todos
#define NUM_OCULTOS 8
#define NUM_SAIDAS 1
// Quantas vezes vou forçar a rede a estudar os dados
#define EPOCAS 20000
#define TAXA_APRENDIZADO 0.1

// Criei esse enum só para facilitar minha vida na hora de acessar o array (em vez de usar números soltos)
enum { I_HP=0, I_MN, I_STR, I_INT, I_DEF, I_MDEF };

// --- FUNÇÃO DE COMPATIBILIDADE ---
// Fiz isso para garantir que meu código rode no Linux (meu pc) e no Windows (faculdade)
void pausar_tela() {
    #ifdef _WIN32
        system("pause");
    #else
        printf("\nPressione ENTER para sair...");
        getchar(); getchar();
    #endif
}

// --- A ESTRUTURA DA MINHA REDE NEURAL (RNN) ---
typedef struct {
    double Wx[NUM_ENTRADAS][NUM_OCULTOS]; // Pesos da Entrada
    double Wh[NUM_OCULTOS][NUM_OCULTOS];  // Pesos da Memória (Aqui está o segredo da RNN)
    double Wy[NUM_OCULTOS][NUM_SAIDAS];   // Pesos da Saída
    
    // Bias para ajudar no ajuste fino da matemática
    double bias_h[NUM_OCULTOS];
    double bias_y[NUM_SAIDAS];
    
    // Aqui eu guardo o histórico do que a rede pensou em cada turno
    double estados_ocultos[SEQ_LEN + 1][NUM_OCULTOS]; 
} RNN;

// --- FUNÇÕES MATEMÁTICAS NECESSÁRIAS ---
double sigmoide(double x) { return 1.0 / (1.0 + exp(-x)); }
double d_sigmoide(double x) { return x * (1.0 - x); }
// Uso tangente hiperbólica para a memória, pois ela lida melhor com números negativos
double tanh_custom(double x) { return tanh(x); }
double d_tanh(double x) { return 1.0 - x * x; }
// Minha função para gerar pesos aleatórios no início
double aleatorio() { return ((double)rand() / RAND_MAX) * 2.0 - 1.0; }

// --- INICIALIZAÇÃO ---
// Antes de começar, preciso preencher a rede com valores aleatórios
void inicializar_rnn(RNN *rede) {
    for (int i = 0; i < NUM_ENTRADAS; i++)
        for (int j = 0; j < NUM_OCULTOS; j++) rede->Wx[i][j] = aleatorio();
    for (int i = 0; i < NUM_OCULTOS; i++)
        for (int j = 0; j < NUM_OCULTOS; j++) rede->Wh[i][j] = aleatorio();
    for (int i = 0; i < NUM_OCULTOS; i++) {
        for (int j = 0; j < NUM_SAIDAS; j++) rede->Wy[i][j] = aleatorio();
        rede->bias_h[i] = aleatorio();
    }
    rede->bias_y[0] = aleatorio();
}

// --- FORWARD PASS (O CÉREBRO FUNCIONANDO) ---
// Aqui a IA "assiste" a luta turno a turno para tomar uma decisão
double forward(RNN *rede, double entrada[SEQ_LEN][NUM_ENTRADAS]) {
    // Limpo a memória inicial
    for(int h=0; h<NUM_OCULTOS; h++) rede->estados_ocultos[0][h] = 0.0;

    // Loop pelos 3 turnos
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int j = 0; j < NUM_OCULTOS; j++) {
            double ativacao = rede->bias_h[j];
            
            // 1. Processo o que está acontecendo AGORA
            for (int i = 0; i < NUM_ENTRADAS; i++) 
                ativacao += entrada[t][i] * rede->Wx[i][j];
            
            // 2. Misturo com o que aconteceu ANTES (Memória Recorrente)
            for (int h = 0; h < NUM_OCULTOS; h++) 
                ativacao += rede->estados_ocultos[t][h] * rede->Wh[h][j];
            
            // Guardo essa memória para o próximo passo
            rede->estados_ocultos[t+1][j] = tanh_custom(ativacao);
        }
    }
    
    // Calculo a probabilidade final de vitória baseada na última memória
    double saida_final = rede->bias_y[0];
    for (int j = 0; j < NUM_OCULTOS; j++) 
        saida_final += rede->estados_ocultos[SEQ_LEN][j] * rede->Wy[j][0];
    
    return sigmoide(saida_final);
}

// --- TREINAMENTO (APRENDENDO COM OS ERROS) ---
// Utilizei o algoritmo BPTT (Backpropagation Through Time)
void treinar(RNN *rede, double entrada[SEQ_LEN][NUM_ENTRADAS], double alvo) {
    double saida_predita = forward(rede, entrada);
    
    // Calculo o erro: O quanto a IA errou?
    double delta_saida = (alvo - saida_predita) * d_sigmoide(saida_predita);
    
    // Ajusto os pesos da saída
    for (int j = 0; j < NUM_OCULTOS; j++) {
        rede->Wy[j][0] += TAXA_APRENDIZADO * delta_saida * rede->estados_ocultos[SEQ_LEN][j];
    }
    rede->bias_y[0] += TAXA_APRENDIZADO * delta_saida;

    // Agora vem a parte complexa: Voltar no tempo para corrigir a memória
    double erro_proximo[NUM_OCULTOS] = {0};
    
    for (int t = SEQ_LEN - 1; t >= 0; t--) {
        double delta_oculto[NUM_OCULTOS];
        for (int j = 0; j < NUM_OCULTOS; j++) {
            // Calculo o erro considerando o futuro e o presente
            double erro = (t == SEQ_LEN - 1) ? (delta_saida * rede->Wy[j][0]) : erro_proximo[j];
            delta_oculto[j] = erro * d_tanh(rede->estados_ocultos[t+1][j]);
            
            // Ajusto todos os pesos (Bias, Entrada e Memória)
            rede->bias_h[j] += TAXA_APRENDIZADO * delta_oculto[j];
            for (int i = 0; i < NUM_ENTRADAS; i++) 
                rede->Wx[i][j] += TAXA_APRENDIZADO * delta_oculto[j] * entrada[t][i];
            for (int h = 0; h < NUM_OCULTOS; h++) 
                rede->Wh[h][j] += TAXA_APRENDIZADO * delta_oculto[j] * rede->estados_ocultos[t][h];
        }
        // Preparo o erro para o passo anterior (passado)
        for (int h = 0; h < NUM_OCULTOS; h++) {
            erro_proximo[h] = 0.0;
            for (int j = 0; j < NUM_OCULTOS; j++) erro_proximo[h] += delta_oculto[j] * rede->Wh[h][j];
        }
    }
}

// --- MEU MOTOR DE JOGO (SIMULAÇÃO) ---
// Criei essa função para automatizar a batalha. Ela gera os dados para a RNN.
void simular_gameplay(double input_rnn[SEQ_LEN][NUM_ENTRADAS], double h[6], double m[6]) {
    
    // Variáveis locais para controlar a vida durante a simulação
    double h_vida = h[I_HP], h_mana = h[I_MN];
    double m_vida = m[I_HP], m_mana = m[I_MN];

    printf("\n--- INICIANDO BATALHA AUTOMATICA ---\n");

    for(int t = 0; t < SEQ_LEN; t++) {
        // Normalizo os dados (divido por 100) para a rede neural conseguir ler
        input_rnn[t][0] = h_vida / 100.0; input_rnn[t][1] = h_mana / 100.0; 
        input_rnn[t][2] = h[I_STR] / 100.0; input_rnn[t][3] = h[I_INT] / 100.0; 
        input_rnn[t][4] = h[I_DEF] / 100.0; input_rnn[t][5] = h[I_MDEF] / 100.0;

        input_rnn[t][6] = m_vida / 100.0; input_rnn[t][7] = m_mana / 100.0; 
        input_rnn[t][8] = m[I_STR] / 100.0; input_rnn[t][9] = m[I_INT] / 100.0; 
        input_rnn[t][10]= m[I_DEF] / 100.0; input_rnn[t][11]= m[I_MDEF] / 100.0;

        // --- MINHA LÓGICA DE COMBATE ---
        
        // 1. Turno do Heroi
        double dano_heroi = 0;
        // Se eu tiver mais inteligencia que força e tiver mana, uso magia!
        if(h[I_INT] > h[I_STR] && h_mana >= 10) {
            dano_heroi = h[I_INT] - m[I_MDEF]; // Ataco a Defesa Mágica
            h_mana -= 10;
            printf("T%d: Usei MAGIA (Int %.0f vs MDef %.0f). ", t+1, h[I_INT], m[I_MDEF]);
        } else {
            dano_heroi = h[I_STR] - m[I_DEF]; // Ataco a Defesa Física
            printf("T%d: Usei ESPADA (Str %.0f vs Def %.0f). ", t+1, h[I_STR], m[I_DEF]);
        }
        
        if(dano_heroi < 0) dano_heroi = 0; // Não posso curar o inimigo batendo nele
        m_vida -= dano_heroi;
        printf("Dano causado: %.0f.\n", dano_heroi);

        // 2. Turno do Monstro
        double dano_monstro = 0;
        // O monstro é simples: 50% de chance de usar magia se tiver mana
        if(m_mana >= 10 && (rand() % 2 == 0)) {
            dano_monstro = m[I_INT] - h[I_MDEF];
            m_mana -= 10;
        } else {
            dano_monstro = m[I_STR] - h[I_DEF];
        }
        
        if(dano_monstro < 0) dano_monstro = 0;
        h_vida -= dano_monstro;

        // Garanto que a vida não fique negativa
        if(h_vida < 0) h_vida = 0;
        if(m_vida < 0) m_vida = 0;
    }
    printf("--- FIM DOS 3 TURNOS ---\n");
    printf("Status Atual -> Heroi HP: %.0f | Monstro HP: %.0f\n", h_vida, m_vida);
}

// --- O MAIN: ONDE TUDO COMEÇA ---
int main() {
    srand(time(0));
    RNN rnn;
    inicializar_rnn(&rnn);

    printf("--- PREPARANDO MINHA REDE NEURAL ---\n");
    
    // Criei dois cenários extremos para a rede aprender o básico
    // Cenário 1: Vitória fácil (Heroi forte vs Monstro fraco)
    double h_win[6] = {100, 50, 90, 20, 80, 50}; 
    double m_lose[6]= {100, 100,20, 80, 10, 80}; 
    double dados_win[SEQ_LEN][NUM_ENTRADAS];
    
    // Cenário 2: Derrota certa (Heroi fraco vs Boss)
    double h_lose[6] = {80, 20, 30, 30, 20, 20};
    double m_win[6]  = {200,100, 90, 90, 80, 80};
    double dados_lose[SEQ_LEN][NUM_ENTRADAS];

    printf("1. Gerando dados de simulacao...\n");
    simular_gameplay(dados_win, h_win, m_lose);
    simular_gameplay(dados_lose, h_lose, m_win);
    
    printf("\n2. Treinando a IA com %d epocas... ", EPOCAS);
    for (int i = 0; i < EPOCAS; i++) {
        treinar(&rnn, dados_win, 1.0);  // Ensino que aqui eu ganho
        treinar(&rnn, dados_lose, 0.0); // Ensino que aqui eu perco
    }
    printf("TREINO CONCLUIDO!\n");

    // --- MODO INTERATIVO ---
    double h_user[6], m_user[6];
    double input_rnn[SEQ_LEN][NUM_ENTRADAS];
    int continuar = 1;

    while(continuar) {
        printf("\n==========================================\n");
        printf("       MEU SIMULADOR: RPG + IA\n");
        printf("==========================================\n");
        
        printf("Defina os status do MEU HEROI:\n");
        printf("[HP] [Mana] [Forca] [Int] [Def] [MDef]: ");
        scanf("%lf %lf %lf %lf %lf %lf", &h_user[0], &h_user[1], &h_user[2], &h_user[3], &h_user[4], &h_user[5]);

        printf("\nDefina os status do INIMIGO:\n");
        printf("[HP] [Mana] [Forca] [Int] [Def] [MDef]: ");
        scanf("%lf %lf %lf %lf %lf %lf", &m_user[0], &m_user[1], &m_user[2], &m_user[3], &m_user[4], &m_user[5]);

        // 1. Rodo a minha simulação de jogo
        simular_gameplay(input_rnn, h_user, m_user);

        // 2. Peço para a rede neural prever o futuro
        double prob = forward(&rnn, input_rnn);
        
        printf("\n>>> O QUE A MINHA IA PENSOU? <<<\n");
        printf("Chance de Vitoria calculada: %.2f%%\n", prob * 100);

        if(prob > 0.7) printf("Conclusao: Estou com a vantagem!\n");
        else if(prob < 0.3) printf("Conclusao: O inimigo e muito forte, vou perder.\n");
        else printf("Conclusao: A luta esta empatada.\n");

        printf("\nTestar de novo? (1-Sim, 0-Nao): ");
        scanf("%d", &continuar);
    }

    return 0;
}