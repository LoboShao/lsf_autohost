#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <curl/curl.h>
#include <unistd.h>
#include "lssched.h"

static const int HANDLER_ID = 113;

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
    int useAutoHost;  /* flag indicating this job uses auto host selection */
    char *selectedHost;  /* ML-selected host (set during match phase) */
} autohost_data;

/* Memory buffer for CURL responses */
struct MemoryStruct {
    char *memory;
    size_t size;
};

/* Function declarations */
static autohost_data *create_autohost_data(void);
static void destroy_autohost_data(void *p);
static char *selectHostML(void *candGroupList, struct jobInfo *jInfo);
static char *selectHostRandom(void *candGroupList);
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);
static char *callMLServer(const char *jsonData);
static int getHostFeatures(void *hostPtr, double *features);

/* Callback functions */
static int autohost_new(void *resreq);
static int autohost_match(autohost_data *data, void *candGroupList, void *reasonTb);
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
    handler->matchFn = (RsrcReqHandler_MatchFn) autohost_match;
    handler->sortFn = (RsrcReqHandler_SortFn) NULL;
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
int sched_match_limit(void *param) { return (0); }
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
        char *pAUTO_HOST = strstr(extsched[0], "AUTO_HOST[");
        
        if (pAUTO_HOST && strchr(pAUTO_HOST, ']')) {
            autohost_data *data;
            char key[MAXLINELEN];
            static int reqId = 0;

            data = create_autohost_data();
            if (data == NULL) {
                ls_syslog(LOG_ERR, "%s: create_autohost_data() failed", fname);
                return (-1);
            }

            data->useAutoHost = 1;
            
            snprintf(key, sizeof(key), "#AUTO_HOST[]#%d#", reqId++);
            
            ls_syslog(LOG_DEBUG, "%s: auto host selection enabled, key=<%s>", fname, key);
            extsched_resreq_setobject(resreq, HANDLER_ID, key, data);
        }
    }

    return (0);
}

// static int autohost_match(autohost_data *data, void *candGroupList, void *reasonTb)
// {
//     static char fname[] = "autohost_match";
//     struct candHostGroup *candGroupEntry = NULL;

//     ls_syslog(LOG_DEBUG, "%s: entering...", fname);

//     if (data == NULL || !data->useAutoHost) {
//         return (0);  /* Not our job */
//     }

//     if (candGroupList == NULL) {
//         return (-1);
//     }

//     /* Select host using ML algorithm */
//     data->selectedHost = selectHostML(candGroupList, NULL);
    
//     if (data->selectedHost == NULL) {
//         ls_syslog(LOG_DEBUG, "%s: no host selected", fname);
//         return (0);
//     }

//     ls_syslog(LOG_INFO, "%s: selected host <%s> for auto host job", fname, data->selectedHost);

//     /* Filter candidate list to only include selected host */
//     candGroupEntry = lsb_cand_getnextgroup(candGroupList);
//     while (candGroupEntry != NULL) {
//         int index = 0;
        
//         while (index < candGroupEntry->numOfMembers) {
//             struct candHost *candHost = NULL;
//             char *hname = NULL, *cname = NULL;

//             candHost = &(candGroupEntry->candHost[index]);
            
//             if (IS_REMOVED_CAND_HOST(candHost->flag)) {
//                 index++;
//                 continue;
//             }

//             if (extsched_getHostID(candHost->hostPtr, &hname, &cname) < 0) {
//                 index++;
//                 continue;
//             }

//             /* Keep only the selected host, remove others */
//             if (strcmp(hname, data->selectedHost) != 0) {
//                 extsched_cand_removehost(candGroupEntry, index);
//                 ls_syslog(LOG_DEBUG, "%s: removed host <%s> from candidates", fname, hname);
//             } else {
//                 index++;  /* Keep this host, move to next */
//             }

//             FREEUP(hname);
//             FREEUP(cname);
//         }
        
//         candGroupEntry = lsb_cand_getnextgroup(NULL);
//     }

//     return (0);
// }

static int autohost_match(autohost_data *data, void *candGroupList, void *reasonTb)
{
    static char fname[] = "autohost_match";
    struct candHostGroup *candGroupEntry = NULL;

    ls_syslog(LOG_DEBUG, "%s: entering...", fname);

    if (data == NULL || !data->useAutoHost) {
        return (0);  /* Not our job */
    }

    if (candGroupList == NULL) {
        return (-1);
    }

    /* Select host using ML algorithm */
    data->selectedHost = selectHostML(candGroupList, NULL);
    
    if (data->selectedHost == NULL) {
        ls_syslog(LOG_DEBUG, "%s: no host selected", fname);
        return (0);
    }

    ls_syslog(LOG_INFO, "%s: selected host <%s> for auto host job", fname, data->selectedHost);

    /* Filter candidate list to only include selected host */
    candGroupEntry = lsb_cand_getnextgroup(candGroupList);
    while (candGroupEntry != NULL) {
        int index = 0;
        
        while (index < candGroupEntry->numOfMembers) {
            struct candHost *candHost = NULL;
            char *hname = NULL, *cname = NULL;

            candHost = &(candGroupEntry->candHost[index]);
            
            if (IS_REMOVED_CAND_HOST(candHost->flag)) {
                index++;
                continue;
            }

            if (extsched_getHostID(candHost->hostPtr, &hname, &cname) < 0) {
                index++;
                continue;
            }

            /* Keep only the selected host, remove others */
            if (strcmp(hname, data->selectedHost) != 0) {
                extsched_cand_removehost(candGroupEntry, index);
                ls_syslog(LOG_DEBUG, "%s: removed host <%s> from candidates", fname, hname);
                continue;
            } else {
                ls_syslog(LOG_DEBUG, "%s: keeping selected host <%s>", fname, hname);
                index++;  /* Keep this host, move to next */
            }

            FREEUP(hname);
            FREEUP(cname);
        }
        
        candGroupEntry = lsb_cand_getnextgroup(NULL);
    }

    return (0);
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

    if (job != NULL) {
        struct jobInfo info = {0};
        int event;

        if (extsched_getJobInfo(job, &info) < 0) {
            ls_syslog(LOG_DEBUG, "%s: extsched_getJobInfo() failed", fname);
            return (0);
        }

        event = extsched_determineEvent(job, alloc, allocLimitList, flag);
        
        ls_syslog(LOG_INFO, "%s: job %d[%d] auto host event=%d on host <%s>", 
                  fname, info.jobId, info.jobIndex, event, 
                  data->selectedHost ? data->selectedHost : "unknown");

        extsched_freeJobInfo(&info);
    }

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
    data->selectedHost = NULL;
    return data;
}

static void destroy_autohost_data(void *p)
{
    autohost_data *data = (autohost_data *)p;
    if (data) {
        FREEUP(data->selectedHost);
        FREEUP(data);
    }
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

/*-------------------------------------------------------------------------
 * Call ML server for host selection
 *-------------------------------------------------------------------------*/
static char *callMLServer(const char *jsonData)
{
    static char fname[] = "callMLServer";
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    struct curl_slist *headers = NULL;
    char *selectedHost = NULL;
    
    chunk.memory = malloc(1);
    chunk.size = 0;
    
    curl = curl_easy_init();
    if(!curl) {
        ls_syslog(LOG_ERR, "%s: curl_easy_init() failed", fname);
        FREEUP(chunk.memory);
        return NULL;
    }
    
    /* Set HTTP headers */
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    /* Set CURL options */
    curl_easy_setopt(curl, CURLOPT_URL, ML_SERVER_URL);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);  /* 5 second timeout */
    
    /* Perform the request */
    res = curl_easy_perform(curl);
    
    if(res != CURLE_OK) {
        ls_syslog(LOG_ERR, "%s: curl_easy_perform() failed: %s", 
                  fname, curl_easy_strerror(res));
    } else {
        /* Parse JSON response to extract selected_host */
        char *hostStart = strstr(chunk.memory, "\"selected_host\":");
        if (hostStart) {
            hostStart += strlen("\"selected_host\":");  /* Move past the key */
            hostStart = strchr(hostStart, '"');  /* Find opening quote of value */
            if (hostStart) {
                hostStart++;  /* Skip the opening quote */
                char *hostEnd = strchr(hostStart, '"');
                if (hostEnd) {
                    size_t len = hostEnd - hostStart;
                    selectedHost = (char *)malloc(len + 1);
                    if (selectedHost) {
                        strncpy(selectedHost, hostStart, len);
                        selectedHost[len] = '\0';
                        ls_syslog(LOG_INFO, "%s: ML server returned host: %s", fname, selectedHost);
                    }
                }
            }
        }
    }
    
    /* Cleanup */
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    FREEUP(chunk.memory);
    
    return selectedHost;
}

/*-------------------------------------------------------------------------
 * Extract host features for ML model
 *-------------------------------------------------------------------------*/
static int getHostFeatures(void *hostPtr, double *features)
{
    static char fname[] = "getHostFeatures";
    
    /* TODO: Replace with actual host metrics extraction
     * For now, using random values for testing */
    
    /* Generate random features between 0 and 1 */
    features[0] = (double)(1 % 100) / 100.0;  /* Simulated CPU usage */
    features[1] = (double)(1 % 100) / 100.0;  /* Simulated memory usage */
    features[2] = (double)(1 % 100) / 100.0;  /* Simulated load average */
    features[3] = (double)(1 % 100) / 100.0;  /* Simulated swap usage */
    
    ls_syslog(LOG_DEBUG, "%s: features=[%.3f, %.3f, %.3f, %.3f]", 
              fname, features[0], features[1], features[2], features[3]);
    
    return 0;
}

/*-------------------------------------------------------------------------
 * ML Host Selection Functions
 *-------------------------------------------------------------------------*/
static char *selectHostML(void *candGroupList, struct jobInfo *jInfo)
{
    static char fname[] = "selectHostML";
    struct candHostGroup *candGroupEntry = NULL;
    char jsonBuffer[JSON_BUFFER_SIZE];
    char *selectedHost = NULL;
    int jsonLen = 0;
    int firstHost = 1;
    
    ls_syslog(LOG_DEBUG, "%s: entering ML host selection", fname);
    
    /* Build JSON request with candidate hosts and their features */
    jsonLen = snprintf(jsonBuffer, sizeof(jsonBuffer), "{\"candidates\": [");
    
    candGroupEntry = lsb_cand_getnextgroup(candGroupList);
    while (candGroupEntry != NULL && jsonLen < sizeof(jsonBuffer) - 1024) {
        int index = 0;
        
        while (index < candGroupEntry->numOfMembers && jsonLen < sizeof(jsonBuffer) - 1024) {
            struct candHost *candHost = &(candGroupEntry->candHost[index]);
            char *hname = NULL, *cname = NULL;
            double features[4] = {0.0, 0.0, 0.0, 0.0};
            
            if (IS_REMOVED_CAND_HOST(candHost->flag)) {
                index++;
                continue;
            }
            
            if (extsched_getHostID(candHost->hostPtr, &hname, &cname) >= 0) {
                /* Get host features */
                if (getHostFeatures(candHost->hostPtr, features) == 0) {
                    /* Add comma separator if not first host */
                    if (!firstHost) {
                        jsonLen += snprintf(jsonBuffer + jsonLen, sizeof(jsonBuffer) - jsonLen, ", ");
                    }
                    firstHost = 0;
                    
                    /* Add host entry to JSON */
                    jsonLen += snprintf(jsonBuffer + jsonLen, sizeof(jsonBuffer) - jsonLen,
                                      "{\"hostname\": \"%s\", \"features\": [%.6f, %.6f, %.6f, %.6f]}",
                                      hname, features[0], features[1], features[2], features[3]);
                }
            }
            
            FREEUP(hname);
            FREEUP(cname);
            index++;
        }
        
        candGroupEntry = lsb_cand_getnextgroup(NULL);
    }
    
    jsonLen += snprintf(jsonBuffer + jsonLen, sizeof(jsonBuffer) - jsonLen, "]}");
    
    ls_syslog(LOG_DEBUG, "%s: JSON request length: %d", fname, jsonLen);
    
    /* Call ML server */
    selectedHost = callMLServer(jsonBuffer);
    
    /* Fallback to random selection if ML server fails */
    if (selectedHost == NULL) {
        ls_syslog(LOG_WARNING, "%s: ML server failed, falling back to random selection", fname);
        selectedHost = selectHostRandom(candGroupList);
    }
    
    return selectedHost;
}

static char *selectHostRandom(void *candGroupList)
{
    static char fname[] = "selectHostRandom";
    struct candHostGroup *candGroupEntry = NULL;
    char **hostList = NULL;
    int hostCount = 0;
    int capacity = 16;
    char *selectedHost = NULL;

    if (candGroupList == NULL) {
        return NULL;
    }

    /* Collect all available hosts */
    hostList = (char **)calloc(capacity, sizeof(char *));
    if (hostList == NULL) {
        return NULL;
    }

    candGroupEntry = lsb_cand_getnextgroup(candGroupList);
    while (candGroupEntry != NULL) {
        int index = 0;
        
        while (index < candGroupEntry->numOfMembers) {
            struct candHost *candHost = &(candGroupEntry->candHost[index]);
            char *hname = NULL, *cname = NULL;

            if (IS_REMOVED_CAND_HOST(candHost->flag)) {
                index++;
                continue;
            }

            if (extsched_getHostID(candHost->hostPtr, &hname, &cname) >= 0) {
                if (hostCount >= capacity) {
                    capacity *= 2;
                    char **newList = (char **)realloc(hostList, capacity * sizeof(char *));
                    if (newList == NULL) {
                        FREEUP(hname);
                        FREEUP(cname);
                        /* Free existing list before returning */
                        for (int i = 0; i < hostCount; i++) {
                            FREEUP(hostList[i]);
                        }
                        FREEUP(hostList);
                        return NULL;
                    }
                    hostList = newList;
                }
                
                hostList[hostCount++] = strdup(hname);
                ls_syslog(LOG_DEBUG, "%s: added host <%s> to selection pool", fname, hname);
            }

            FREEUP(hname);
            FREEUP(cname);
            index++;
        }
        
        candGroupEntry = lsb_cand_getnextgroup(NULL);
    }
    
    /* Random selection */
    if (hostCount > 0) {
        int selectedIndex = rand() % hostCount;
        selectedHost = strdup(hostList[selectedIndex]);
        ls_syslog(LOG_INFO, "%s: randomly selected host <%s> from %d candidates", 
                  fname, selectedHost, hostCount);
    }

    /* Cleanup */
    for (int i = 0; i < hostCount; i++) {
        FREEUP(hostList[i]);
    }
    FREEUP(hostList);

    return selectedHost;
}