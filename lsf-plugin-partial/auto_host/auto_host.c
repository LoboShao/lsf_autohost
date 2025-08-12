#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <curl/curl.h>
#include <unistd.h>
#include "lssched.h"
#include "lsf.h"
#include "lsbatch.h"

static const int HANDLER_ID = 118;

#ifndef MAXLINELEN
#define MAXLINELEN 512
#endif

#ifndef FREEUP
#define FREEUP(p) { if ( p ) { free(p); (p) = NULL; } }
#endif


/* ML Server configuration */
#define ML_SERVER_URL "http://localhost:5000/select_host"
#define JSON_BUFFER_SIZE 65536

/* Plugin data structure for AUTO_HOST requests */
typedef struct {
    int useAutoHost;
    double *input_vector;
    int vector_length;
    double job_cores;
    double job_mem;
} autohost_data;

/* Memory buffer for CURL responses */
struct MemoryStruct {
    char *memory;
    size_t size;
};



/* Function declarations */
static autohost_data *create_autohost_data(void);
static void destroy_autohost_data(void *p);
static int extract_job_cores_from_resreq(void *resreq);
static int extract_job_memory_from_resreq(void *resreq);
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);
static char *callMLServer(const char *jsonData);
int parseMLResponse(const char *jsonResponse, double *scores, int count);
static int buildInputVector(autohost_data *data, void *candGroupList, void *job);
static double getHostResourceRatio(struct candHost *host, const char *resource_type);


/* Callback functions */
static int autohost_new(void *resreq);
static int autohost_sort(autohost_data *data, void *candGroupList, void *reasonTb);
static void autohost_free(autohost_data *data);
static int autohost_checkAlloc(autohost_data *data, void *job, void *alloc, void *allocLimitList);
static int autohost_notifyAlloc(autohost_data *data, void *job, void *alloc, void *allocLimitList, int flag);

/*-------------------------------------------------------------------------
 * Plugin framework functions
 *-------------------------------------------------------------------------*/
int sched_version(void *param)
{
    return (0);
}

int sched_init(void *param)
{
    static char fname[] = "autohost_sched_init";
    RsrcReqHandlerType *handler = NULL;
    
    ls_syslog(LOG_DEBUG, "%s: entering...", fname);
    
    handler = (RsrcReqHandlerType *)calloc(1, sizeof(RsrcReqHandlerType));
    if (handler == NULL) {
        ls_syslog(LOG_ERR, "%s: calloc() failed", fname);
        return (-1);
    }

    handler->newFn = (RsrcReqHandler_NewFn) autohost_new;
    handler->freeFn = (RsrcReqHandler_FreeFn) autohost_free;
    handler->matchFn = (RsrcReqHandler_MatchFn) NULL;

    handler->sortFn = (RsrcReqHandler_SortFn) autohost_sort;
    handler->notifyAllocFn = (RsrcReqHandler_NotifyAllocFn) autohost_notifyAlloc;
    handler->checkAllocFn = (RsrcReqHandler_CheckAllocFn) autohost_checkAlloc;

    extsched_resreq_registerhandler(HANDLER_ID, handler);
    ls_syslog(LOG_INFO, "%s: Handler registered with ID %d", fname, HANDLER_ID);

    FREEUP(handler);
    
    /* Initialize CURL globally */
    curl_global_init(CURL_GLOBAL_ALL);
    
    srand(time(NULL));  /* Initialize random seed */
    return (0);
}


int sched_pre_proc(void *param) { return (0); }


int sched_match_limit(void *param) 
{
    /* For now, just return 0 - host pruning will be handled in autohost_sort */
    return (0);
}
int sched_order_alloc(void *param) { return (0); }
int sched_post_proc(void *param) { return (0); }

int sched_finalize(void *param) 
{ 
    /* Cleanup CURL */
    curl_global_cleanup();
    return (0); 
}

/*-------------------------------------------------------------------------
 * Callback handler functions
 *-------------------------------------------------------------------------*/
static int autohost_new(void *resreq)
{
    static char fname[] = "autohost_new";
    char **extsched;
    int num = 0;

    ls_syslog(LOG_DEBUG, "%s: entering...", fname);

    if (resreq == NULL) {
        return (0);
    }

    extsched = extsched_resreq_getextresreq(resreq, &num);


    if (extsched == NULL || num <= 0) {
        return (0);
    }

    if (num > 1 && extsched[0]) {
        char *pAUTO_HOST = strstr(extsched[0], "AUTO_HOST");
        
        if (pAUTO_HOST) {
            autohost_data *data;
            char key[MAXLINELEN];
            static int reqId = 0;

            data = create_autohost_data();
            if (data == NULL) {
                ls_syslog(LOG_ERR, "%s: create_autohost_data() failed", fname);
                return (-1);
            }

            data->useAutoHost = 1;
            
            /* Try to extract job cores from resreq memory structure */
            int extracted_cores = extract_job_cores_from_resreq(resreq);
            if (extracted_cores > 0) {
                data->job_cores = (double)extracted_cores;
                ls_syslog(LOG_INFO, "%s: extracted job cores from resreq: %.1f", fname, data->job_cores);
            }
            
            /* Try to extract job memory from resreq memory structure */
            int extracted_memory = extract_job_memory_from_resreq(resreq);
            if (extracted_memory > 0) {
                data->job_mem = (double)extracted_memory;
                ls_syslog(LOG_INFO, "%s: extracted job memory from resreq: %.1f MB", fname, data->job_mem);
            }
            
    
            if (strchr(pAUTO_HOST, '[')) {
                char *bracket_start = strchr(pAUTO_HOST, '[');
                char *bracket_end = strchr(bracket_start, ']');
                if (bracket_end) {
                    char req_str[256];
                    int len = bracket_end - bracket_start - 1;
                    if (len > 0 && len < sizeof(req_str)) {
                        strncpy(req_str, bracket_start + 1, len);
                        req_str[len] = '\0';
                        
                        /* Parse cores,memory format - only override if not extracted from resreq */
                        char *comma = strchr(req_str, ',');
                        if (comma) {
                            *comma = '\0';
                            if (extracted_cores <= 0) {
                                data->job_cores = strtod(req_str, NULL);
                            }
                            if (extracted_memory <= 0) {
                                data->job_mem = strtod(comma + 1, NULL);
                            }
                            ls_syslog(LOG_INFO, "%s: AUTO_HOST fallback - cores: %.1f, memory: %.1f MB", 
                                      fname, data->job_cores, data->job_mem);
                        }
                    }
                }
            }
            
            /* Log final values used */
            ls_syslog(LOG_INFO, "%s: final job requirements - cores: %.1f, memory: %.1f MB", 
                      fname, data->job_cores, data->job_mem);
            
            snprintf(key, sizeof(key), "#AUTO_HOST#%d#", reqId++);
            
            ls_syslog(LOG_DEBUG, "%s: auto host selection enabled, key=<%s>", fname, key);
            extsched_resreq_setobject(resreq, HANDLER_ID, key, data);
        }
    }

    return (0);
}





static int autohost_sort(autohost_data *data, void *candGroupList, void *reasonTb)
{
    static char fname[] = "autohost_sort";
    struct candHostGroup *candGroupEntry = NULL;

    ls_syslog(LOG_DEBUG, "%s: entering...", fname);

    if (data == NULL || !data->useAutoHost) {
        ls_syslog(LOG_DEBUG, "%s: skipping - not an autohost job", fname);
        return 0;
    }

    if (candGroupList == NULL) {
        ls_syslog(LOG_DEBUG, "%s: candidate group list is NULL", fname);
        return -1;
    }

    candGroupEntry = lsb_cand_getnextgroup(candGroupList);
    if (candGroupEntry == NULL) {
        ls_syslog(LOG_DEBUG, "%s: candidate group is NULL", fname);
        return -3;
    }

    /* Log available information for debugging */
    ls_syslog(LOG_DEBUG, "%s: candidate group info - maxOfMembers: %d, numOfMembers: %d", 
              fname, candGroupEntry->maxOfMembers, candGroupEntry->numOfMembers);


    /* Build input vector dynamically based on hosts and job requirements */
    if (buildInputVector(data, candGroupList, NULL) != 0) {
        ls_syslog(LOG_ERR, "%s: buildInputVector failed", fname);
        return -4;
    }

    /* Build JSON for ML server */
    char jsonBuffer[JSON_BUFFER_SIZE];
    int pos = snprintf(jsonBuffer, sizeof(jsonBuffer), "{\"input_vector\": [");
    
    for (int i = 0; i < data->vector_length; i++) {
        if (i > 0) {
            pos += snprintf(jsonBuffer + pos, sizeof(jsonBuffer) - pos, ", ");
        }
        pos += snprintf(jsonBuffer + pos, sizeof(jsonBuffer) - pos, "%.6f", data->input_vector[i]);
    }
    pos += snprintf(jsonBuffer + pos, sizeof(jsonBuffer) - pos, "]}");

    ls_syslog(LOG_DEBUG, "%s: JSON payload: %s", fname, jsonBuffer);

    char *jsonResponse = callMLServer(jsonBuffer);
    if (!jsonResponse) {
        ls_syslog(LOG_DEBUG, "%s: ML server returned NULL response", fname);
        return -2;
    }

    int numCandidates = candGroupEntry->numOfMembers;
    double *scores = malloc(sizeof(double) * numCandidates);
    if (!scores) {
        ls_syslog(LOG_DEBUG, "%s: memory allocation failed for scores", fname);
        FREEUP(jsonResponse);
        return -4;
    }

    int parsed = parseMLResponse(jsonResponse, scores, numCandidates);
    if (parsed != numCandidates) {
        ls_syslog(LOG_DEBUG, "%s: parseMLResponse error: parsed %d, expected %d", fname, parsed, numCandidates);
        free(scores);
        FREEUP(jsonResponse);
        return -5;
    }

    ls_syslog(LOG_DEBUG, "%s: successfully parsed %d scores", fname, parsed);

    /* Sort hosts by ML scores (bubble sort) */
    for (int i = 0; i < numCandidates - 1; i++) {
        for (int j = 0; j < numCandidates - 1 - i; j++) {
            if (scores[j] < scores[j + 1]) {
                struct candHost temp = candGroupEntry->candHost[j];
                candGroupEntry->candHost[j] = candGroupEntry->candHost[j + 1];
                candGroupEntry->candHost[j + 1] = temp;
                
                double tempScore = scores[j];
                scores[j] = scores[j + 1];
                scores[j + 1] = tempScore;
            }
        }
    }

    free(scores);
    FREEUP(jsonResponse);
    return 0;
}


static void autohost_free(autohost_data *data)
{
    static char fname[] = "autohost_free";
    ls_syslog(LOG_DEBUG, "%s: entering...", fname);
    destroy_autohost_data((void *)data);
}

static int autohost_checkAlloc(autohost_data *data, void *job, void *alloc, void *allocLimitList)
{
    static char fname[] = "autohost_checkAlloc";
    
    if (data == NULL || !data->useAutoHost) {
        return (0);
    }

    ls_syslog(LOG_DEBUG, "%s: checking allocation for auto host job", fname);
    /* No additional checks needed for auto host selection */
    return (0);
}

static int autohost_notifyAlloc(autohost_data *data, void *job, void *alloc, void *allocLimitList, int flag)
{
    static char fname[] = "autohost_notifyAlloc";
    
    if (data == NULL || !data->useAutoHost) {
        return (0);
    }
    ls_syslog(LOG_DEBUG, "%s: checking allocation for auto host job", fname);
    return (0);
}
/*-------------------------------------------------------------------------
 * Helper functions
 *-------------------------------------------------------------------------*/
static autohost_data *create_autohost_data(void)
{
   autohost_data *data = (autohost_data *)calloc(1, sizeof(autohost_data));
   if (data == NULL) {
       return NULL;
   }
   data->useAutoHost = 0;
   data->input_vector = NULL;
   data->vector_length = 0;
   data->job_cores = 0.0;
   data->job_mem = 0.0;
   return data;
}

static void destroy_autohost_data(void *p)
{
   autohost_data *data = (autohost_data *)p;
   if (data) {
       if (data->input_vector) {
           FREEUP(data->input_vector);
       }
       FREEUP(data);
   }
}


/*
 * Extract job core requirements from resreq memory structure
 * Based on observed pattern: cores are stored at memory offset 2 (Min and Max)
 */
static int extract_job_cores_from_resreq(void *resreq)
{
    static char fname[] = "extract_job_cores_from_resreq";
    int *int_ptr;
    int cores = 0;
    
    if (resreq == NULL) {
        ls_syslog(LOG_DEBUG, "%s: resreq is NULL", fname);
        return 0;
    }
    
    /* Cast resreq to int array and extract cores from position 2 */
    int_ptr = (int *)resreq;
    cores = int_ptr[2];

    return cores;
}

/*
 * Extract job memory requirements from resreq memory structure
 */
static int extract_job_memory_from_resreq(void *resreq)
{
    static char fname[] = "extract_job_memory_from_resreq";
    extsched_rsrcReqInfo *rsrcReqInfo = NULL;
    int memoryMB = 0;
    
    if (!resreq) {
        ls_syslog(LOG_ERR, "%s: NULL resreq parameter", fname);
        return 0;
    }
    rsrcReqInfo = extsched_getRsrcReqInfo((INT_RsrcReq *)resreq);
    if (!rsrcReqInfo) {
        ls_syslog(LOG_DEBUG, "%s: No resource requirement info available", fname);
        return 0;
    }
    
    /* Iterate through resource consumption requirements */
    for (int i = 0; i < rsrcReqInfo->nRsrcConsump; i++) {
        if (rsrcReqInfo->rsrcConsump[i].rsrcName) {
            ls_syslog(LOG_DEBUG, "%s: Found resource: %s, amount: %.2f", 
                     fname, rsrcReqInfo->rsrcConsump[i].rsrcName,
                     rsrcReqInfo->rsrcConsump[i].amount);
            
            /* Check for memory-related resource names */
            if (strcmp(rsrcReqInfo->rsrcConsump[i].rsrcName, "mem") == 0 ||
                strcmp(rsrcReqInfo->rsrcConsump[i].rsrcName, "memory") == 0 ||
                strcmp(rsrcReqInfo->rsrcConsump[i].rsrcName, "maxmem") == 0) {
                
                memoryMB = (int)rsrcReqInfo->rsrcConsump[i].amount;
                ls_syslog(LOG_INFO, "%s: Extracted memory requirement: %d MB", 
                         fname, memoryMB);
                break;
            }
        }
    }
    
    /* Clean up allocated memory */
    extsched_freeRsrcReqInfo(&rsrcReqInfo);
    
    return memoryMB;
}


/*-------------------------------------------------------------------------
 * CURL callback for receiving data
 *-------------------------------------------------------------------------*/
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if(ptr == NULL) {
        /* out of memory! */
        ls_syslog(LOG_ERR, "WriteMemoryCallback: not enough memory (realloc returned NULL)");
        return 0;
    }

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

static int buildInputVector(autohost_data *data, void *candGroupList, void *job)
{
    static char fname[] = "buildInputVector";
    struct candHostGroup *candGroupEntry = NULL;
    int num_hosts = 0;
    
    ls_syslog(LOG_DEBUG, "%s: entering...", fname);

    candGroupEntry = lsb_cand_getnextgroup(candGroupList);
    if (candGroupEntry == NULL) {
        ls_syslog(LOG_ERR, "%s: no candidate groups found", fname);
        return -1;
    }

    num_hosts = candGroupEntry->numOfMembers;
    if (num_hosts <= 0) {
        ls_syslog(LOG_ERR, "%s: no candidate hosts found", fname);
        return -2;
    }

    /* Vector length: 2 values per host (core_ratio, mem_ratio) + 2 job values (cores, mem) */
    data->vector_length = num_hosts * 2 + 2;
    data->input_vector = (double *)calloc(data->vector_length, sizeof(double));
    if (data->input_vector == NULL) {
        ls_syslog(LOG_ERR, "%s: memory allocation failed for input vector", fname);
        return -3;
    }

    /* Use job requirements if already set, otherwise use defaults */
    if (data->job_cores <= 0.0) {
        data->job_cores = 4.0;  /* Default: 4 cores */
    }
    if (data->job_mem <= 0.0) {
        data->job_mem = 2048.0; /* Default: 2048MB memory */
    }
    
    ls_syslog(LOG_DEBUG, "%s: using job requirements - cores: %.1f, memory: %.1f MB", 
              fname, data->job_cores, data->job_mem);

    /* Build host resource ratios */
    for (int i = 0; i < num_hosts; i++) {
        struct candHost *host = &candGroupEntry->candHost[i];
        
        /* Get host resource ratios */
        double core_ratio = getHostResourceRatio(host, "cores");
        double mem_ratio = getHostResourceRatio(host, "memory");
        
        /* Store in input vector: [h1_core_ratio, h1_mem_ratio, h2_core_ratio, h2_mem_ratio, ...] */
        data->input_vector[i * 2] = core_ratio;
        data->input_vector[i * 2 + 1] = mem_ratio;
        
        char *hostname = NULL;
        char *clustername = NULL;
        if (extsched_getHostID(host->hostPtr, &hostname, &clustername) == 0) {
            // ls_syslog(LOG_DEBUG, "%s: host[%d] %s: core_ratio=%.3f, mem_ratio=%.3f", 
            //           fname, i, hostname, core_ratio, mem_ratio);
            if (hostname) free(hostname);
            if (clustername) free(clustername);
        } else {
            ls_syslog(LOG_DEBUG, "%s: host[%d] <unknown>: core_ratio=%.3f, mem_ratio=%.3f", 
                      fname, i, core_ratio, mem_ratio);
        }
    }

    /* Add job requirements to the end of vector */
    data->input_vector[num_hosts * 2] = data->job_cores;
    data->input_vector[num_hosts * 2 + 1] = data->job_mem;

    ls_syslog(LOG_DEBUG, "%s: input vector built with %d elements", fname, data->vector_length);
    return 0;
}

static double getHostResourceRatio(struct candHost *host, const char *resource_type)
{
    static char fname[] = "getHostResourceRatio";
    
    if (host == NULL || resource_type == NULL) {
        ls_syslog(LOG_ERR, "%s: invalid parameters", fname);
        return 0.0;
    }

    char *hostname = NULL;
    char *clustername = NULL;
    double ratio = 0.0;
    double available = 0.0;
    double maximum = 0.0;

    /* Get hostname for logging */
    if (extsched_getHostID(host->hostPtr, &hostname, &clustername) != 0) {
        ls_syslog(LOG_ERR, "%s: failed to get host ID", fname);
        return 0.0;
    }

    /* Use extsched_host_resources() to get resource information */
    struct hostResources *hostRes = extsched_host_resources(host->hostPtr);
    if (hostRes != NULL) {
        /* Find available and maximum resource values */
        for (int i = 0; i < hostRes->nres; i++) {
            struct resources *res = &hostRes->res[i];
            
            if (res->cType != FLOAT_VAL) continue;
            
            if (strcmp(resource_type, "cores") == 0) {
                // /* Look for CPU/core related resources */
                // if (strcmp(res->resName, "ncpus") == 0) {
                //     maximum = res->val.fval;
                // } else if (strcmp(res->resName, "cpu") == 0) {
                //     available = res->val.fval;
                // } else if (strcmp(res->resName, "r15s") == 0) {
                //     /* Load average - calculate available as ncpus - load */
                //     if (maximum > 0.0) {
                //         available = maximum - res->val.fval;
                //         if (available < 0.0) available = 0.0;
                //     }
                /* Use ncpus for maximum, slots for available */
                if (strcmp(res->resName, "ncpus") == 0) {
                    maximum = res->val.fval;
                } else if (strcmp(res->resName, "slots") == 0) {
                    available = res->val.fval;
                }
            } else if (strcmp(resource_type, "memory") == 0) {
                /* Look for memory related resources */
                if (strcmp(res->resName, "maxmem") == 0) {
                    maximum = res->val.fval;
                } else if (strcmp(res->resName, "mem") == 0) {
                    available = res->val.fval;
                }
            }
        }
        
        /* Calculate ratio if we found both values */
        if (available > 0.0 && maximum > 0.0) {
            ratio = available / maximum;
            ls_syslog(LOG_DEBUG, "%s: host %s %s ratio: %.3f/%.3f = %.3f", 
                      fname, hostname, resource_type, available, maximum, ratio);
        } else {
            /* Log available resources for debugging */
            ls_syslog(LOG_DEBUG, "%s: host %s available resources (avail=%.3f, max=%.3f):", 
                      fname, hostname, available, maximum);
            for (int i = 0; i < hostRes->nres; i++) {
                struct resources *res = &hostRes->res[i];
                if (res->cType == FLOAT_VAL) {
                    ls_syslog(LOG_DEBUG, "%s:   %s = %.3f", fname, res->resName, res->val.fval);
                } else if (res->cType == STRING_VAL) {
                    ls_syslog(LOG_DEBUG, "%s:   %s = %s", fname, res->resName, res->val.sval);
                }
            }
        }
    } else {
        ls_syslog(LOG_DEBUG, "%s: host %s - extsched_host_resources returned NULL", fname, hostname);
    }

    if (hostname) free(hostname);
    if (clustername) free(clustername);
    return ratio;
}


/*-------------------------------------------------------------------------
 * ML Host Selection Functions
 *-------------------------------------------------------------------------*/
static char *callMLServer(const char *jsonData)
{
    static char fname[] = "callMLServer";
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    struct curl_slist *headers = NULL;

    chunk.memory = malloc(1);  // will be grown as needed by realloc
    chunk.size = 0;

    curl = curl_easy_init();

    if (!curl) {
        fprintf(stderr, "%s: Failed to initialize CURL\n", fname);
        return NULL;
    }

    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, ML_SERVER_URL);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "%s: curl_easy_perform() failed: %s\n", fname, curl_easy_strerror(res));
        FREEUP(chunk.memory);
        chunk.memory = NULL;
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    return chunk.memory;  // WARNING: must not FREEUP here if returning it
}

int parseMLResponse(const char *jsonResponse, double *scores, int count)
{
    char *scores_start = strstr(jsonResponse, "\"scores\":[");
    if (!scores_start) return -1;
    
    scores_start += 10; // Skip "scores":["
    
    for (int i = 0; i < count; i++) {
        scores[i] = strtod(scores_start, &scores_start);
        if (*scores_start == ',') scores_start++; // Skip comma
    }
    
    return count;
}
