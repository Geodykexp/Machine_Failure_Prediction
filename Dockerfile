FROM public.ecr.aws/lambda/python:3.12

# Install uv
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# Set working directory to Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies into the system site-packages for Lambda
RUN uv pip install --system --requirement pyproject.toml

# Copy application files
COPY lambda_function.py ./
COPY machine_failure_prediction.pkl ./

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]