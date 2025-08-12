#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

#define JSON_BUFFER_SIZE 8192
#define ML_SERVER_URL "http://localhost:5000/select_host"

#define FREEUP(p) do { if (p) free(p); p = NULL; } while(0)


/* ---------------------------
 * Custom candidate structures
 * --------------------------- */
typedef struct {
    char *hostname;
    double features[4];  // Example: [CPU, Mem, GPU, Avail]
    int removed;         // 0 = valid, 1 = removed
} SimpleCandHost;

typedef struct {
    int num_cores;
    int memory_request;
} Job;

typedef struct {
    SimpleCandHost *hosts;
    int numHosts;
} SimpleCandGroup;

/* Memory buffer for CURL responses */
struct MemoryStruct {
    char *memory;
    size_t size;
};


/*-------------------------------------------------------------------------
 * CURL callback for receiving data
 *-------------------------------------------------------------------------*/
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    char *ptr = realloc(mem->memory, mem->size + realsize + 1);

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

static char *callMLServer(const char *jsonData)
{
    static char fname[] = "callMLServer";
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    struct curl_slist *headers = NULL;

    chunk.memory = malloc(1);  // will be grown as needed by realloc
    chunk.size = 0;

    curl_global_init(CURL_GLOBAL_ALL);
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
    } else {
        // Print full response
        printf("Response from server: %s\n", chunk.memory);
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    curl_global_cleanup();

    return chunk.memory;  // WARNING: must not FREEUP here if returning it
}



/* ---------------------------
 * Prepare JSON string only
 * --------------------------- */
void prepareJson(SimpleCandGroup *group, Job *job, char *jsonBuffer, size_t bufferSize) {
    int jsonLen = 0;
    int firstHost = 1;
    jsonLen = snprintf(jsonBuffer, bufferSize, 
        "{\"job\": {\"cores\": %d, \"memory\": %d}, \"candidates\": [", job->num_cores, job->memory_request);

    for (int i = 0; i < group->numHosts && jsonLen < bufferSize - 1024; i++) {
        SimpleCandHost *host = &group->hosts[i];

        if (host->removed)
            continue;

        if (!firstHost) {
            jsonLen += snprintf(jsonBuffer + jsonLen, bufferSize - jsonLen, ", ");
        }
        firstHost = 0;

        jsonLen += snprintf(jsonBuffer + jsonLen, bufferSize - jsonLen,
            "{\"hostname\": \"%s\", \"features\": [%.6f, %.6f, %.6f, %.6f]}",
            host->hostname, host->features[0], host->features[1],
            host->features[2], host->features[3]);
    }

    jsonLen += snprintf(jsonBuffer + jsonLen, bufferSize - jsonLen, "]}");
}

int main() {
    // Global libcurl init
    curl_global_init(CURL_GLOBAL_ALL);

    SimpleCandHost hosts[] = {
        { "host1", {0.1, 0.2, 0.3, 0.4}, 0 },
        { "host2", {0.5, 0.6, 0.7, 0.8}, 0 },
        { "host3", {0.9, 1.0, 1.1, 1.2}, 1 }  // removed
    };

    Job job = {.num_cores = 4, .memory_request = 8192 };
    SimpleCandGroup group = { hosts, 3 };
    char jsonBuffer[JSON_BUFFER_SIZE];

    prepareJson(&group, &job, jsonBuffer, sizeof(jsonBuffer));
    printf("Prepared JSON:\n%s\n", jsonBuffer);

    char *returns = callMLServer(jsonBuffer);
    if (returns) {
        printf("ML server returns score list: %s\n", returns);
        free(returns);
    } else {
        printf("ML server did not return a valid score list.\n");
    }

    // Global libcurl cleanup
    curl_global_cleanup();

    return 0;
}
