FROM vllm/vllm-openai:latest

WORKDIR /app

COPY serve.sh /app/serve.sh
RUN chmod +x /app/serve.sh

EXPOSE 8000

ENTRYPOINT ["/app/serve.sh"]
