# Patent Project
This is a Patent project code named "Patent Guru". Its designed to help patent applicants and organizations with Tech Transfer decisions, patent application checks and building/writing patents

**Monitoring**

The project is designed to monitor the container metrics for the patent docker container. Data scrapping is done using prometheus. Replace the existing docker engine to scrap the metrics to Prometheus

PRE REQUISITE - Grafana and Prometheus docker image should be available
STEP 1
**Docker Engine** - Windows Docker Desktop>Settings - The metrics-addr parameter does the container metrics scrapping
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "metrics-addr": "127.0.0.1:9323"
}

STEP 2
USe the prometheus file (available in repo) to send the scrapped metrics to Prometheus. Remember to use the ipconfig command to get the latest IP. It does not work with localhost as reference

STEP 3 
Run the patent app docker
Run the prometheus docker
Run the grafana docker

Plot graph in grafana, set alerts etc

Run Prometheus (local mount of prometheus.yml) - command (use IP from ipconfig)
docker run --name prometheus -d -v /path/to/prometheus.yml:\prometheus.yml -p 172.29.224.1:9090:9090 prom/prometheus

