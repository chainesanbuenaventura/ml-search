# environment variables
export MLFLOW_TRACKING_URI="http://40.112.217.252:5000/"
export AZURE_STORAGE_ACCESS_KEY="MMt5nOGV4MSdF0FOaohg5kIXNXfF3X6Kny5rNQC69/s8u5qx0jyRk+h9K/xT7f824wmmWHlQQ4YQz66GvuMSzA=="

# change owner of node modules folder to 
sudo chown -R $USER /usr/local/lib/node_modules

# install elasticdump
npm install elasticdump -g

# mlflow
pip install mlflow==1.8.0