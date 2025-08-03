FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
CMD ["marimo", "edit", "your_notebook.py", "--host", "0.0.0.0", "--no-token"]
