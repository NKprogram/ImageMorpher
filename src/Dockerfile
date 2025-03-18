FROM public.ecr.aws/lambda/python:3.10

# Install system-level dependencies, including the one for libGL and for fonts
RUN yum -y update \
    && yum install -y ffmpeg libSM libXext mesa-libGL dejavu-sans-mono-fonts \
    && yum clean all

# Set Matplotlib configuration directory to a writable location
ENV MPLCONFIGDIR=/tmp/matplotlib

# Copy requirements.txt into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy your application code
COPY app/* ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "main.handler" ]
