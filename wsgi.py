from waitress import serve
from main import app
import os
from werkzeug.middleware.proxy_fix import ProxyFix

# Add ProxyFix middleware
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    print("Starting server on port", port)
    # Development mode
    if os.getenv("FLASK_ENV") == "development":
        app.run(host="0.0.0.0", port=port, debug=True)
    # Production mode
    else:
        serve(
            app,
            host="0.0.0.0",
            port=port,
            threads=int(os.getenv("WAITRESS_THREADS", 4)),
            url_scheme='https',
            channel_timeout=900,  # 15 minutes timeout
            cleanup_interval=30,
            max_request_body_size=1073741824  # 1GB
        )
