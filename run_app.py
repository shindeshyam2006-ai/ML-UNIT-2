"""One-command launcher for CIE-2 Activity 2: Simple House Price Prediction Studio."""

import os
import uvicorn


def main() -> None:
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    print(f"\n  Activity 2: Simple House Price Prediction Studio")
    print(f"  Open in browser: http://{host}:{port}\n")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
