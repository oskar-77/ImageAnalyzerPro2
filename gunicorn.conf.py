# Gunicorn configuration for improved performance
bind = "0.0.0.0:5000"
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 60  # Reduced timeout
keepalive = 2
max_requests = 100
max_requests_jitter = 10
preload_app = True
reload = True

# Optimization settings
worker_tmp_dir = "/dev/shm"