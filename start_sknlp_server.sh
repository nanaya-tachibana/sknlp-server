#!/bin/bash
# check resouce limits
uvicorn server:app \
--host 0.0.0.0 \
--port $HTTP_PORT \
--no-access-log \
--log-level warning \
--timeout-keep-alive 600 \
--workers 1