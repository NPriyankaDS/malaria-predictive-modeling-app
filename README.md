# Mapping Seagrass app

## Docker

**Create the Image**
```
docker build -t omdena_app .
```

**Create the Container**
```
docker run -p 8501:8501 omdena_app
```

Go to ```http://localhost:8501/``` and visualize the app:

![Home View](./src/home_view.PNG)