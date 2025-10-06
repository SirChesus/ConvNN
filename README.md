MyProject/
├── CNN.py
├── load_data.py
├── model.pth              
├── java_backend/
│   ├── pom.xml
│   └── src/
│       └── main/
│           ├── java/
│           │   └── com/example/cnnapp/
│           │       ├── CnnAppApplication.java
│           │       ├── controller/UploadController.java
│           │       ├── service/PredictService.java
│           │       └── util/ImagePreprocessor.java
│           └── resources/
│               ├── templates/index.html
│               └── static/style.css




-----CNN.py-----
Conv2D -> creates grids of number, higher values = stronger activations (detection of edges/texture, etc.)
    - in_channels = number of channels in the input image (1 - Greyscale, 3 - RGB)
    - out_channels = number of filters (feature maps) the layer learns
    - kernel_size = size of the conv filter (n x n) or tuple: (h x w)
    - padding = adds pixels around the input

MaxPool2d -> downsamples feature maps by sliding kernel (n x n) over them and for each window replacing w/ single maximum value
    - kernel_size = size of the window over what the max is taken

EXAMPLE:

Feature Map:
    1  3  2  4
    5  6  1  2
    7  2  8  3
    4  9  2  1

MaxPool2d(2) -> 2x2 kernels 
   
    | 1  3 | 2  4
    | 5  6 | 1  2
      7  2   8  3
      4  9   2  1

RESULTS:
    6  4
    9  8

-----java_backend-----  
pom.xml (Maven Project Object Model)
    - top-level XML setup ->
    - parent config -> Inherits default configs from the offical Spring Boot parent project
        - spring-boot-starter-parent provides default plugin versions, pre-configured compiler settings, standard dependency managment
        (empty relative path means look up online, not locally)

    -project metadata -> defines project identity & build type
        - <groupId> -> organization or namespace (reverse domain name) -> com.example
        - <artifactId> -> name of project's artifact -> JAR filename (cnnapp)
        - <version> -> project version #
        - <packaging> -> specifies build output type:
            - jar for backend apps
            - war for deplyable web archives
        
    -properties -> defines variables used thru the file

    -dependencies
        - spring-boot-starter-web -> Needed to start a web server
            - adds Spring MVC (controllers, RESR API's)
            - includes embedded Tomcat server
            - Handles incomping HTTP requests
        - spring-boot-starter-thymeleaf -> Needed to render .html
            - Adds Thymeleaf, Spring's html template engine
            - Lets you use ${variable} expressions from properties in .html
            - Automatically loads templates from templates\ 
        -onnxruntime -> needed to load & run CNN model
            - provides OrtEnviroment, OrtSession, OnnxTensor
    
    -build section -> without this need to manually handle classpaths & run configs
        - packages dependencies+code into one runnable JAR
        - Enables mvn spring-boot:run
        - adds startup hooks (@SpringBootApplication)





