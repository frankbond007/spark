Steps:

1. Build the base spark image using the following command:

docker build -t spark_base:latest .

2. Run 

docker-compose up --build

If you want to scale up the number of workers use the following command instead:

docker-compose up --scale worker={number_of_desired_workers} --build
