/* Inference for GGUF Qwen-3 models in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/mman.h>

// ----------------------------------------------------------------------------
// Transformer model
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size
    int seq_len; // max sequence length
    int head_dim; // attention dimension
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms in each layer
    float* rms_att_weight; // (layer, dim)
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, n_heads * head_dim)
    float* wk; // (layer, dim, n_kv_heads * head_dim)
    float* wv; // (layer, dim, n_kv_heads * head_dim)
    float* wo; // (layer, n_heads * head_dim, dim)
    float* wq_norm; // (layer, head_dim)
    float* wk_norm; // (layer, head_dim)
    // weights for ffn. w1 = up, w3 = gate, w2 = down
    float* w1; // (layer, dim, hidden_dim)
    float* w2; // (layer, hidden_dim, dim)
    float* w3; // (layer, dim, hidden_dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // Same as token_embedding_table. GGUF has the final layer anyway
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    // TODO remove the unused
    float* x; // activation at current time stamp (dim,)
    float* xb; // buffer (dim,)
    float* xb2; // an additional buffer just for convenience (dim,)
    float* xb3; // an additional buffer just for convenience (att_head_dim,)
    float* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q;   // query (att_head_dim,)
    float* k; // key (dim,)
    float* v; // value (dim,)
    float* att; // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)


    float *batch_x;
    float *batch_xb;
    float *batch_xb2;
    float *batch_xb3;
    float *batch_q;
    float *batch_k;
    float *batch_v;
    float *batch_att;
    float *batch_hb;
    float *batch_hb2;

} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    int att_head_dim = p->n_heads * p->head_dim; 
    int kv_dim = p->n_kv_heads * p->head_dim;  // 1024  

    // TODO -remove the unused
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));  
    s->xb2 = calloc(p->dim, sizeof(float));  
    s->xb3 = calloc(att_head_dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(att_head_dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));  
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));

    // Allocate batch tensors (assuming these are added to RunState or allocated here)
    // For simplicity, assuming single token output - could be extended for full batch output
    // temp context windows = 8192
    int c_win = 8192;
    s->batch_x = (float *)malloc(c_win * p->dim * sizeof(float));
    s->batch_xb = (float *)malloc(c_win * p->dim * sizeof(float));
    s->batch_xb2 = (float *)malloc(c_win * p->dim * sizeof(float));
    s->batch_xb3 = (float *)malloc(c_win * att_head_dim * sizeof(float));
    s->batch_q = (float *)malloc(c_win * att_head_dim * sizeof(float));
    s->batch_k = (float *)malloc(c_win * kv_dim * sizeof(float));
    s->batch_v = (float *)malloc(c_win * kv_dim * sizeof(float));
    s->batch_att = (float *)malloc(c_win * p->n_heads * p->seq_len * sizeof(float));
    s->batch_hb = (float *)malloc(c_win * p->hidden_dim * sizeof(float));
    s->batch_hb2 = (float *)malloc(c_win * p->hidden_dim * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->xb3 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->xb3);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);

    // Cleanup batch tensors
    free(s->batch_x);
    free(s->batch_xb);
    free(s->batch_xb2);
    free(s->batch_xb3);
    free(s->batch_q);
    free(s->batch_k);
    free(s->batch_v);
    free(s->batch_att);
    free(s->batch_hb);
    free(s->batch_hb2);
}

// Map GGUF layers to transformer weights
void memory_map_weights(TransformerWeights* w, Config* p, void* pt) {
    unsigned long long n_layers = p->n_layers;
    float *ptr = (float*) pt; 

    w->wcls = ptr;   // last layer in TR
    ptr += p->vocab_size * p->dim;
    w->rms_final_weight = ptr; // right before the last
    ptr += p->dim;
    w->token_embedding_table = ptr; // first layer  
    ptr += p->vocab_size * p->dim;
    w->wk = ptr;
    ptr += p->dim * (p->n_kv_heads * p->head_dim);  // 1024 x 1024 = dim (1024) x  num_kv_heads (8) x p->head_dim (128)
    w->wk_norm = ptr;
    ptr += p->head_dim; //head_dim (128)
    w->rms_att_weight = ptr;
    ptr += p->dim; //dimension (1024)
    w->wo = ptr;
    ptr += (p->n_heads * p->head_dim) * p->dim;  // attention heads (16) x head dim (128) * dim
    w->wq = ptr;
    ptr += p->dim * (p->n_heads * p->head_dim);
    w->wq_norm = ptr;
    ptr += p->head_dim; //head_dim (128)     
    w->wv = ptr;
    ptr += p->dim * (p->n_kv_heads * p->head_dim); // equal to wk
    w->w2 = ptr;
    ptr += p->hidden_dim * p->dim; //ffn.down 3072 *1024 
    w->w3 = ptr;
    ptr += p->dim * p->hidden_dim; // ffn.gate
    w->rms_ffn_weight = ptr;
    ptr += p->dim;                 // ffn.norm
    w->w1 = ptr;
    ptr += p->dim * p->hidden_dim; //ffn.up
}

// --------------------------------------
// read GGUF
void read_checkpoint(char *checkpoint, Config *config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = (ssize_t)ftell(file); // get the file size, in bytes
    fclose(file);
    
    //printf("file size is %zd", *file_size);

    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    
    void* weights_ptr = ((char*)*data) + 5951648; // skip header bytes. header_size = 5951648 TODO
                                             // gguf total header = file size - (last tensor size + last offset)          
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer *t, char *checkpoint_path) {
    // read in the Weights from the GGUF
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    if (t->data && t->data != MAP_FAILED) {
        munmap(t->data, t->file_size);
        t->data = NULL;
    }
}

// load the GGUF config file
void load_config(Transformer *t) {
    FILE *f = fopen("header.txt", "r");
    if (!f) {perror("Failed to open header.txt"); exit(1);}

    char line[512];
    int line_num = 0;

    while (fgets(line, sizeof(line), f)) {
        line_num++;

        char *cfg = strtok(line, " ");
        while (cfg) {
            char *eq = strchr(cfg, '=');
            if (!eq) {
                //fprintf(stderr, "Warning: malformed cfg on line %d: %s\n", line_num, cfg);
                cfg = strtok(NULL, " ");
                continue;
            }

            *eq = '\0';
            char *key = cfg;
            char *val = eq + 1;

            // Match keys
            if (strcmp(key, "QWEN3_EMBEDDING_LENGTH") == 0) {
                t->config.dim = atoi(val);
            } else if (strcmp(key, "QWEN3_FEED_FORWARD_LENGTH") == 0) {
                t->config.hidden_dim = atoi(val);
            } else if (strcmp(key, "QWEN3_BLOCK_COUNT") == 0) {
                t->config.n_layers = atoi(val);
            } else if (strcmp(key, "QWEN3_ATTENTION_HEAD_COUNT") == 0) {
                t->config.n_heads = atoi(val);
            } else if (strcmp(key, "QWEN3_ATTENTION_HEAD_COUNT_KV") == 0) {
                t->config.n_kv_heads = atoi(val);
            } else if (strcmp(key, "QWEN3_CONTEXT_LENGTH") == 0) {
                t->config.seq_len = atoi(val);
            } else if (strcmp(key, "QWEN3_ATTENTION_KEY_LENGTH") == 0) {
                t->config.head_dim = atoi(val);
            }else if (strncmp(key, "TOKENIZER_GGML_TOKENS", 22) == 0) { 
                char *len_ptr = strstr(val, "ARRAY_LENGTH=");  //handling nested config
                if (len_ptr) {
                    t->config.vocab_size = atoi(len_ptr + strlen("ARRAY_LENGTH="));
                }
            }
            cfg = strtok(NULL, " ");
        }
    }
    fclose(f);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer 
void rmsnorm(float *o, float *x, float *weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


void batch_rmsnorm(float *batch_o, float *batch_x, float *weight, int size, int num_tokens) {
    for (int b = 0; b < num_tokens; b++) {
        float *o = batch_o + b * size;
        float *x = batch_x + b * size;

        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-6f;
        ss = 1.0f / sqrtf(ss);

        // normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}

void batch_matmul(float *batch_xout, float *batch_x, float *w, int n, int d, int num_tokens) {
    // Process all tokens in batch
    for (int b = 0; b < num_tokens; b++) {
        float *xout = batch_xout + b * d;
        float *x = batch_x + b * n;

        // W (d,n) @ x (n,) -> xout (d,)
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }
}

float *forward_batch(Transformer *transformer, int *tokens, int num_tokens, int start_pos, int past_tokens)
{
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    
    int kv_dim = p->n_kv_heads * p->head_dim;  
    int kv_mul = p->n_heads / p->n_kv_heads; 
    int att_head_dim = p->n_heads * p->head_dim; 

    int layer_offset = 62923776/4; // offset to the GGUF next layer for the same tensor type TODO

    // num_tokens = total number of tokens in one go, including batch process     
    // For prefix caching
    // past_tokens is 0, when 1) it is the inital conversation, or 
    // 2) it is the generation phase.
 
    // When there are past tokens, only the rest (prompts) are processed.
    if (past_tokens > 0) {
        num_tokens = num_tokens - past_tokens;
    }

    // Copy token embeddings for all tokens in batch
    // In prefix caching, only new prompts and
    // generated tokens are processed and save KV caches
    for (int b = 0; b < num_tokens; b++) {
        memcpy(s->batch_x + b * p->dim,
               w->token_embedding_table + tokens[b + past_tokens] * p->dim,
               p->dim * sizeof(float)); // past_tokens for prefix caching
    }

    // Forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        int loff = l * p->seq_len * kv_dim;

        // Process batch through attention rmsnorm
        batch_rmsnorm(s->batch_xb, s->batch_x, w->rms_att_weight + l * layer_offset, p->dim, num_tokens);

        // Batch matrix multiplications for Q, K, V
        batch_matmul(s->batch_q, s->batch_xb, w->wq + l * layer_offset, p->dim, att_head_dim, num_tokens);
        batch_matmul(s->batch_k, s->batch_xb, w->wk + l * layer_offset, p->dim, kv_dim, num_tokens);
        batch_matmul(s->batch_v, s->batch_xb, w->wv + l * layer_offset, p->dim, kv_dim, num_tokens);

        // Apply batch RMSNorm to all query heads
        batch_rmsnorm(s->batch_q, s->batch_q, w->wq_norm + l * layer_offset, p->head_dim, num_tokens * p->n_heads);

        // Apply batch RMSNorm to all key heads
        batch_rmsnorm(s->batch_k, s->batch_k, w->wk_norm + l * layer_offset, p->head_dim, num_tokens * p->n_kv_heads);

        // Apply rotary position encoding
        // num_tokens is 1 in the generation phase
        for (int b = 0; b < num_tokens; b++)
        {
            // start_pos is the postion mark of batch or generation.
            // Encode the RoPE except the cached tokens.
            // TODO - Reduce variables
            int pos;
            if (past_tokens > 0) {
                pos = past_tokens + start_pos + b;
            }
            else {
                pos = start_pos + b;
            }


            // Process all query heads
            for (int h = 0; h < p->n_heads; h++)
            {
                float *q = s->batch_q + b * att_head_dim + h * p->head_dim;

                // Apply rotary position encoding to query head
                for (int i = 0; i < p->head_dim / 2; i++)
                {
                    float freq = 1.0f / powf(1000000.0f, (float)i / (p->head_dim / 2));
                    float fcr = cosf(pos * freq);
                    float fci = sinf(pos * freq);

                    float x_q = q[i];
                    float y_q = q[i + p->head_dim / 2];
                    q[i] = x_q * fcr - y_q * fci;
                    q[i + p->head_dim / 2] = x_q * fci + y_q * fcr;
                }
            }

            // Process key heads (only up to n_kv_heads)
            for (int h = 0; h < p->n_kv_heads; h++)
            {
                float *k = s->batch_k + b * kv_dim + h * p->head_dim;

                // Apply rotary position encoding to key head
                for (int i = 0; i < p->head_dim / 2; i++)
                {
                    float freq = 1.0f / powf(1000000.0f, (float)i / (p->head_dim / 2));
                    float fcr = cosf(pos * freq);
                    float fci = sinf(pos * freq);

                    float x_k = k[i];
                    float y_k = k[i + p->head_dim / 2];
                    k[i] = x_k * fcr - y_k * fci;
                    k[i + p->head_dim / 2] = x_k * fci + y_k * fcr;
                }
            }
        }

        // Update KV cache for all tokens in batch 
        // Do not save caches if multi_turn
        for (int b = 0; b < num_tokens; b++) {
            int pos;
            if (past_tokens > 0) {
                pos = past_tokens + start_pos + b;
            }
            else {
                pos = start_pos + b;
            }
            float* k_cache = s->key_cache + loff + pos * kv_dim;
            float* v_cache = s->value_cache + loff + pos * kv_dim;

            memcpy(k_cache, s->batch_k + b * kv_dim, kv_dim * sizeof(float));
            memcpy(v_cache, s->batch_v + b * kv_dim, kv_dim * sizeof(float));
        }
        
        // Multihead attention with causal masking
        for (int b = 0; b < num_tokens; b++) {
            int current_pos;
            if (past_tokens > 0) {
                current_pos = past_tokens + start_pos + b;
            }
            else {
                current_pos =  start_pos + b;
            }
            #pragma omp parallel for
            for (int h = 0; h < p->n_heads; h++) {
                // Get the query vector for this head and batch item
                float *q = s->batch_q + b * att_head_dim + h * p->head_dim;
                // Attention scores for this head
                float *att = s->batch_att + b * p->n_heads * p->seq_len + h * p->seq_len;

                // Iterate over all timesteps up to current position (causal masking)
                for (int t = 0; t <= current_pos; t++) {
                    // Get the key vector for this head and timestep
                    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;                
                    
                    // Calculate the attention score as dot product of q and k
                    float score = 0;
                    for (int i = 0; i < p->head_dim; i++)
                        score += q[i] * k[i];
                    // Save the score to the attention buffer
                    att[t] = score / sqrtf(p->head_dim);
                }

                // Softmax the scores to get attention weights, from 0..current_pos inclusively
                softmax(att, current_pos + 1);

                // Weighted sum of the values, store back into batch_xb3
                float *xb3 = s->batch_xb3 + b * att_head_dim + h * p->head_dim;
                memset(xb3, 0, p->head_dim * sizeof(float));
                
                for (int t = 0; t <= current_pos; t++) {
                    // Get the value vector for this head and timestep
                    float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;                
                    // Get the attention weight for this timestep
                    float a = att[t];
                    // Accumulate the weighted value into xb3
                    for (int i = 0; i < p->head_dim; i++)
                        xb3[i] += a * v[i];
                }
            }
        }

        // Output projection for batch
        batch_matmul(s->batch_xb2,
                     s->batch_xb3,
                     w->wo + l * layer_offset,
                     att_head_dim, p->dim,
                     num_tokens);

        // Residual connection back into batch_x
        for (int b = 0; b < num_tokens; b++) {
            for (int i = 0; i < p->dim; i++)
                s->batch_x[b * p->dim + i] += s->batch_xb2[b * p->dim + i];
        }

        // FFN rmsnorm for batch
        batch_rmsnorm(s->batch_xb,
                      s->batch_x,
                      w->rms_ffn_weight + l * layer_offset,
                      p->dim,
                      num_tokens);

        // FFN matrix multiplications for batch
        batch_matmul(s->batch_hb,
                     s->batch_xb,
                     w->w1 + l * layer_offset,
                     p->dim, p->hidden_dim,
                     num_tokens);
        batch_matmul(s->batch_hb2,
                     s->batch_xb,
                     w->w3 + l * layer_offset,
                     p->dim, p->hidden_dim,
                     num_tokens);

        // SwiGLU non-linearity for batch
        for (int b = 0; b < num_tokens; b++) {
            for (int i = 0; i < p->hidden_dim; i++) {
                float val = s->batch_hb2[b * p->hidden_dim + i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0f / (1.0f + expf(-val)));
                // elementwise multiply with w3(x)
                val *= s->batch_hb[b * p->hidden_dim + i];
                s->batch_hb2[b * p->hidden_dim + i] = val;
            }
        }

        // Final FFN output for batch
        batch_matmul(s->batch_xb,
                     s->batch_hb2,
                     w->w2 + l * layer_offset,
                     p->hidden_dim,
                     p->dim,
                     num_tokens);

        // Residual connection for batch
        for (int b = 0; b < num_tokens; b++) {
            for (int i = 0; i < p->dim; i++)
                s->batch_x[b * p->dim + i] += s->batch_xb[b * p->dim + i];
        }
    }

    // For simplicity, return logits for the last token in batch
    // Copy last token's hidden state to original state buffer
    memcpy(s->x, s->batch_x + (num_tokens - 1) * p->dim, p->dim * sizeof(float));

    // Final rmsnorm and logits computation
    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);
    matmul(s->logits, s->x, w->wcls, p->dim, p->vocab_size);



    return s->logits;
}

// ----------------------------------------------------------------------------
// TOKENIZER
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Tokenizer;

// ========== GLOBALS ==========

// TODO
#define MAX_VOCAB 151936
#define MAX_MERGES 151386 
#define MAX_TOKENS 1024
#define MAX_TOKEN_LEN 32

char *vocab[MAX_VOCAB];
int num_vocab = 0;

typedef struct {
    char left[MAX_TOKEN_LEN];
    char right[MAX_TOKEN_LEN];
    char merged[MAX_TOKEN_LEN * 2];
    int rank;
} MergeRule;

MergeRule merges[MAX_MERGES];
int merge_count = 0;

unsigned int byte_to_unicode[256];
char unicode_bytes[256][5]; 

//special token
const char *special_tokens[] = {
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>"
};
int num_special_tokens = sizeof(special_tokens) / sizeof(special_tokens[0]);

typedef struct {
    int *data;         // token ID array
    size_t size;       // number of tokens stored
    size_t capacity;   // total allocated capacity
} TokenBuffer;

// ========== LOAD VOCAB ==========
void load_vocab(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    char line[512];  // large enough to hold most tokens safely
    int line_num = 0;

    while (fgets(line, sizeof(line), f)) {
        line_num++;

        // Remove trailing newline / carriage return
        line[strcspn(line, "\r\n")] = '\0';

        // Skip empty lines
        if (line[0] == '\0') continue;

        if (num_vocab >= MAX_VOCAB) {
            fprintf(stderr, "Error: vocab exceeds MAX_VOCAB (%d)\n", MAX_VOCAB);
            break;
        }

        vocab[num_vocab] = strdup(line);
        if (!vocab[num_vocab]) {
            fprintf(stderr, "Error: strdup failed on line %d\n", line_num);
            exit(1);
        }
        num_vocab++;
    }
    fclose(f);
}

void free_vocab() {
    for (int i = 0; i < num_vocab; i++) {
        free(vocab[i]);  // Free each string
        vocab[i] = NULL; 
    }
    num_vocab = 0;
}

// ========== LOAD MERGES ==========
void load_merges(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("merges.txt"); exit(1); }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || isspace(line[0])) continue;
        char *a = strtok(line, " \t\r\n");
        char *b = strtok(NULL, " \t\r\n");
        if (!a || !b) continue;
        strncpy(merges[merge_count].left, a, MAX_TOKEN_LEN);
        strncpy(merges[merge_count].right, b, MAX_TOKEN_LEN);
        snprintf(merges[merge_count].merged, sizeof(merges[merge_count].merged), "%s%s", a, b);
        merges[merge_count].rank = merge_count;
        merge_count++;
    }
    fclose(f);
}

bool pair_equals(const char *a, const char *b, const char *left, const char *right) {
    return strcmp(a, left) == 0 && strcmp(b, right) == 0;
}

int get_merge_rank(const char *left, const char *right) {
    for (int i = 0; i < merge_count; i++) {
        if (pair_equals(left, right, merges[i].left, merges[i].right)) {
            return merges[i].rank;
        }
    }
    return INT_MAX;
}

// ========== BYTE-TO-UNICODE ==========
void init_byte_unicode_map() {
    int n = 0;    
    for (int b = 0; b < 256; b++) {
        // party for gpt-2 based tokenizer
        if ((b >= 33 && b <= 126) ||    // ASCII
            (b >= 161 && b <= 172) ||   // Latin-1 
            (b >= 174 && b <= 255)) {   // Latin-1 except 173
            byte_to_unicode[b] = b;
        } else {
            byte_to_unicode[b] = 256 + n++;  // Map out of Unicode range
        }
        
        // Direct UTF-8 encoding
        uint32_t cp = byte_to_unicode[b];
        if (cp < 128) {                  // 1-byte UTF-8
            unicode_bytes[b][0] = cp;
            unicode_bytes[b][1] = '\0';
        } 
        else if (cp < 2048) {            // 2-byte UTF-8
            unicode_bytes[b][0] = 0xC0 | (cp >> 6);
            unicode_bytes[b][1] = 0x80 | (cp & 0x3F);
            unicode_bytes[b][2] = '\0';
        } 
        else {                           // 3-byte UTF-8
            unicode_bytes[b][0] = 0xE0 | (cp >> 12);
            unicode_bytes[b][1] = 0x80 | ((cp >> 6) & 0x3F);
            unicode_bytes[b][2] = 0x80 | (cp & 0x3F);
            unicode_bytes[b][3] = '\0';
        }
    }
}

// TODO. Returns -1 if not a special token, else index into vocab[]
int match_special_token(const char *str, int *match_len) {  
    for (int i = 0; i < num_special_tokens; i++) {
        const char *tok = special_tokens[i];
        int len = strlen(tok);
        if (strncmp(str, tok, len) == 0) {
            // Now check if this token exists in vocab
            for (int j = 0; j < 151936; j++) { 
                if (strcmp(vocab[j], tok) == 0) {
                    *match_len = len;
                    return j; // return token ID
                }
            }
        }
    }   
    return -1;
}

void build_tokenizer(Tokenizer* t) {
    load_vocab("vocab.txt");
    load_merges("merges.txt");
    init_byte_unicode_map();
}    
 
// build buffer 
void build_token_buffer(TokenBuffer *tb, size_t initial_capacity) {
    tb->data = malloc(initial_capacity * sizeof(int));
    if (!tb->data) { perror("malloc"); exit(1); }
    tb->size = 0;
    tb->capacity = initial_capacity;
}

void free_token_buffer(TokenBuffer *tb) {
    free(tb->data);
    tb->data = NULL;
    tb->size = 0;
    tb->capacity = 0;
}
void append_tokens(TokenBuffer *tb, const int *tokens, size_t n) {
    if (tb->size + n > tb->capacity) {
        while (tb->size + n > tb->capacity) tb->capacity *= 2;
        tb->data = realloc(tb->data, tb->capacity * sizeof(int));
        if (!tb->data) { perror("realloc"); exit(1); }
    }
    memcpy(&tb->data[tb->size], tokens, n * sizeof(int));
    tb->size += n;
}

void encode(Tokenizer* t, char* rendered_prompt, int* prompt_tokens, int* num_prompt_tokens, int multi_turn) {
    if (rendered_prompt == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
    char *tokens[MAX_TOKENS];
    int count = 0;

    const char *p = rendered_prompt;

    while (*p) {
        int match_len = 0;
        int special_id = match_special_token(p, &match_len);
        if (special_id >= 0) {
            // Special token found → store its string
            tokens[count++] = strdup(vocab[special_id]);  // Store as string, not ID
            p += match_len;
            continue;
        }

        // Not a special token → convert byte to unicode
        unsigned char b = *p++;
        tokens[count++] = strdup(unicode_bytes[b]);
    }

    // === BPE merge ===
    bool changed = true;
    while (changed) {
        int best_rank = INT_MAX;
        int best_pos = -1;

        for (int i = 0; i < count - 1; i++) {
            int rank = get_merge_rank(tokens[i], tokens[i + 1]);
            if (rank < best_rank) {
                best_rank = rank;
                best_pos = i;
            }
        }

        if (best_pos == -1) break;

        // if either token is a special token, skip merge 
        if (tokens[best_pos][0] == '<' && strchr(tokens[best_pos], '|') &&  
            tokens[best_pos + 1][0] == '<' && strchr(tokens[best_pos + 1], '|')) {
            break; // don't merge special tokens
        }

        // Merge
        char *merged = malloc(MAX_TOKEN_LEN * 2);
        snprintf(merged, MAX_TOKEN_LEN * 2, "%s%s", tokens[best_pos], tokens[best_pos + 1]);
        free(tokens[best_pos]);
        free(tokens[best_pos + 1]);
        tokens[best_pos] = merged;
        for (int i = best_pos + 1; i < count - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        count--;
        changed = true;
    }

    // === Map tokens to token IDs ===
    int token_ids[MAX_TOKENS];
    int token_id_count = 0;

    for (int i = 0; i < count; i++) {
        int id = -1;
        for (int j = 0; j < 151936; j++) {
            if (strcmp(tokens[i], vocab[j]) == 0) {
                id = j;
                break;
            }
        }

        if (id == -1) {
            fprintf(stderr, "Token not found in vocab: [%s]\n", tokens[i]);
        } else {
            token_ids[token_id_count++] = id;
            //printf("[%s] → id = %d\n", tokens[i], id);
        }
    // TODO
    for (int i = 0; i < token_id_count; i++) {
        prompt_tokens[i] = token_ids[i];
    }
        *num_prompt_tokens = token_id_count;
    }
}
 
// ========== Decoding logic==========
// Inverse byte-to-unicode: utf8 string → byte value (0–255)
int unicode_to_byte(const char *utf8) {
    for (int b = 0; b < 256; b++) {
        if (strcmp(utf8, unicode_bytes[b]) == 0)
            return b;
    }
    return -1; // not found
}

char *decode_token_id(int token_id) {
    const char *encoded = vocab[token_id];
    char *out = malloc(1024);
    int rlen = 0;
    for (int i = 0; encoded[i]; ) {
        int matched = 0;
        for (int len = 1; len <= 3 && encoded[i + len - 1]; len++) {
            char utf8[5] = {0};
            strncpy(utf8, &encoded[i], len);
            int b = unicode_to_byte(utf8);
            if (b >= 0) {
                out[rlen++] = (char)b;
                i += len;
                matched = 1;
                break;
            }
        }
        if (!matched) break;
    }
    out[rlen] = '\0';
    return out;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) 
{
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time
long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; 
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* cli_user_prompt, char* cli_system_prompt, int think_on, int multi_turn, int tps, int ttft, TokenBuffer* tb) {
    // buffers for reading the system prompt and user prompt from stdin
    char system_prompt[512];
    char user_prompt[8192]; 
    char rendered_prompt[8192]; 
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int* )malloc(8192 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    double timer = -1.0;   // TPS timer start
    long timer2 = -1;  // TTFT timer start
    long t_ttft = 0;       // TTFT 
    int count = 0; // decoded token
    int prompt_batch = 1; // prompt or multi-turn aggregation
    int past_tokens = 0; // total prev convesation tokens count for prefix caching



    while (1) { 
        if (user_turn) {
            if (pos == 0){
                read_stdin("Enter system prompt (or Enter to skip): ", system_prompt, sizeof(system_prompt));
            }
            read_stdin("Q: ", user_prompt, sizeof(user_prompt));
            // terminate if user enters a blank prompt
            if (!user_prompt[0]) {
                break;
            }
            // TTFT starts
            if (ttft && timer2 == -1) {timer2 = time_in_ms();}

            // render user/system prompts for Qwen3 
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
                sprintf(rendered_prompt, user_template, user_prompt);
            }

            if (!think_on) {
                strcat(rendered_prompt, "<think>\n\n</think>\n");
            }

            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens, multi_turn);
            pos = 0; // reset the user index
            user_turn = 0;  
            // Add prompt tokens first to the convesation history buffer, tb
            // for multi-turn conversations
            if (multi_turn) {
                append_tokens(tb, prompt_tokens, num_prompt_tokens);
                for (size_t i = 0; i < tb->size; i++) {
                }
                printf("\n");
            }
            printf("A: ");
        }
        
        // For batch processing logic, 2 phases are set. Prompts
        // or past conversations pass as a batch. Generated token 
        // passes one by one.
        // PROMPT PROCESSING PHASE: Use batch processing for prompt
        // or multi-turn aggregation
        int batch_tokens = (multi_turn ? tb->size : num_prompt_tokens);
        if (prompt_batch) {
            int *tokens_to_process = (multi_turn) ? tb->data : prompt_tokens;
            int total_prompt_tokens = batch_tokens;
            //int remaining_tokens = total_prompt_tokens - pos;

            // prefix caching for multiturn. if 0, it's a single-turn
            //tb->size == num_prompt_tokens, then it's the first turn or single-turn.
            // past_tokens counts the number of KV-cached tokens.
            // Only new prompts do the full pass through transformer layers
            past_tokens = (multi_turn ? tb->size - num_prompt_tokens : 0);
            // Process all prompt tokens and/or prev conversations in one batch
            float* logits = forward_batch(transformer, tokens_to_process, total_prompt_tokens, pos, past_tokens);
            next = sample(sampler, logits);
            // Switch to the generation phase
            pos = batch_tokens; // Move position to end of prompt
            prompt_batch = 0;   
        }
        else {
            // GENERATION PHASE: Process one generated token at a time
            // cache_mode = 2 (generation), then is should be saved to the cache and use all the previous kv values
            //past_tokens=0 for prefix caching in generation phase
            float* logits = forward_batch(transformer, &next, 1, pos, 0);            
            next = sample(sampler, logits);
            pos++;
        }

        //printf("num_prompt_tokens: %d \n", num_prompt_tokens);
        // decoding and printing
        if (pos >= batch_tokens) {
            if (multi_turn) {
                append_tokens(tb, &next, 1);
            }
            
            if (next == 151645) { // EOS token ID - TODO
                printf("\n");
                user_turn = 1; prompt_batch = 1;

                // TPS
                if (tps) {
                    fprintf(stderr, "tok/s: %f\n", count / (double)(time_in_ms() - timer) * 1000);
                    timer = -1; count = 0;
                }
                // TTFT
                if (ttft) {
                    fprintf(stderr, "TTFT: %ld ms\n", t_ttft);
                    timer2 = -1; t_ttft = 0;
                }
            }
            else {
                char *decoded = decode_token_id(next);
                printf("%s", decoded);
                fflush(stdout);
                free(decoded);
                if (ttft && t_ttft==0) {t_ttft = time_in_ms() - timer2;}
                if (tps) {
                    count += 1;
                    // timer starts after the first token generation
                    if (timer == -1.0) {timer = time_in_ms();}
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// CLI
void error_usage() {
    fprintf(stderr, "Usage:   run <FP32 GGUF file> [options]\n");
    fprintf(stderr, "Example: ./run Qwen3-0.6B-FP32.gguf\n");  
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -m <int>    multi-turn: 0 = off (defualt), 1 = on\n");
    fprintf(stderr, "  -k <int>    reasoning: 0 = off (defualt), 1 = on\n");
    fprintf(stderr, "  -r <int>    TPS: 0 = off (defualt), 1 = on\n");
    fprintf(stderr, "  -f <int>    ttFt: 0 = off (defualt), 1 = on\n");
    exit(EXIT_FAILURE);
}


int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    float temperature = 0.6f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.95f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int multi_turn = 0;  // multi-turn conversation
    int think_on = 0;    //  reasoning on
    int tps = 0;         // TPS
    int ttft = 0;        // TTFT

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'm') {if ((argv[i+1][0] == '0' || argv[i+1][0] == '1') && argv[i+1][1] == '\0') {
        multi_turn = argv[i+1][0] - '0';} else { error_usage(); } }
        else if (argv[i][1] == 'k') {if ((argv[i+1][0] == '0' || argv[i+1][0] == '1') && argv[i+1][1] == '\0') {
        think_on = argv[i+1][0] - '0';} else { error_usage(); } }
        else if (argv[i][1] == 'r') {if ((argv[i+1][0] == '0' || argv[i+1][0] == '1') && argv[i+1][1] == '\0') {
        tps = argv[i+1][0] - '0';} else { error_usage(); } }
        else if (argv[i][1] == 'f') {if ((argv[i+1][0] == '0' || argv[i+1][0] == '1') && argv[i+1][1] == '\0') {
        ttft = argv[i+1][0] - '0';} else { error_usage(); } }        
        else { error_usage(); }        
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // read config
    Transformer transformer;
    load_config(&transformer);

    // build the Transformer via the GGUF file
    build_transformer(&transformer, checkpoint_path);  

    // build the Tokenizer 
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer); 

    // multi-turn buffer
    TokenBuffer tb;
    build_token_buffer(&tb, 1024);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    printf("Multi-turn = %s, thinKing = %s, tps(R) = %s, ttFt = %s, Temperature = %.2f, top-P = %.2f\n", multi_turn ? "on" : "off", think_on ? "on" : "off", tps ? "on" : "off", ttft ? "on" : "off", temperature, topp);
    printf("Press Enter to exit the chat\n");

    // run!
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, think_on, multi_turn, tps, ttft, &tb); 


    // memory and file handles cleanup
    free_sampler(&sampler); 
    free_vocab();
    free_token_buffer(&tb);
    free_transformer(&transformer);
    return 0;
}
