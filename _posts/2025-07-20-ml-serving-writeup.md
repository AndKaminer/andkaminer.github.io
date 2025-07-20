---
title: Building a Simple Machine Learning Model Serving API
date: 2025-07-20 01:00:00 -0600
categories: [Projects, Machine Learning]
tags: [writeup, ml]
pin: true
---

This month I was planning on working on [FIXable](https://github.com/AndKaminer/FIXable), my rudimentary FIX engine. I got a little bit distracted wrapping up a library of common machine learning algorithms implemented in numpy. I figured building them would help me really internalize how they work. At TTD, I've been working a lot with [Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html), so I thought it might be interesting to build my own little endpoint to serve the models. Thus, this project really started. I'll talk a little about the design decisions behind the project, how it works, and things I learned.


### Design Decisions
My initial infrastructure plan was to build a server docker image with GitHub Actions, push it to AWS ECR, then use that image to run an ECS application. I could have done that, but it turned out to be simpler to just deploy straight to Elastic Beanstalk. I used an LLM to nudge me in the right direction when writing the terraform, which accelerated things somewhat. I still do deployments automatically through GitHub actions. I've been writing plenty of GitLab CI/CD at TTD, so this easy enough.

I chose to use Flask as the base for my server, as it's lightweight (and the only framework I know). It was my first time actually deploying one of these to a prod environment, so I got to learn a little about WSGI. I ended up using Gunicorn to serve the application. No complaints.

If you aren't using Poetry to manage your Python dependencies, you should lock in. It's wonderful, and simplifies a lot of previously annoying things. I highly recommend! That said, the dependencies for this project are very light. It takes like 20 seconds to install all of my dependencies, which is a nice escape from the hell of building dependencies at my real job. 

### How does it work?
The actual serving part of the project is done through a model directory class. This directory allows you to register models, list them, and run inference on data with them. Training of models takes place at registration time. When you register a model, you provide it with a set of arguments. These arguments include data and labels (at least, for models that need to be trained), among other hyperparameters. The model is trained and registered in the model directory. At some point, it may be wise to have an actual database layer for permanent storage of models. Until then, I just store them in memory.

All models are built with basic NumPy arrays. They definitely aren't incredibly robust or performant, but they are at least correct, according to the algorithm design, in most cases.

### The Actual API Schemas
Below, I've included the Pydantic API schemas and routes, should you want to call the API for some reason. I definitely wouldn't recommend doing so in a commercial (or even personal) setting.


```
/predict: 
POST
class InferenceInput(BaseModel):
  model_type: str
  model_id: int = 0
  features: List[float]

/register
POST
class RegisterInput(BaseModel)
  model_type: str
  model_id: int
  kwargs: Dict[str, Any]

/list
POST
class ListInput(BaseModel)
  model_type: str

/listtypes
GET
```

### What did I learn?
I really enjoyed bridging the gap between research formulas and actual implementation details. Thinking about implementation is what makes a good researcher, and thinking about the theory makes a good engineer. Without both, you're going to be left in the dark somewhere, so building from scratch like this was a lot of fun. 
I learned decent bit about service development as well. In the workplace, a lot of the infra is abstracted away, hidden behind layers of YAML files, so building infra from raw terraform and GH Actions was great fun. Abstraction can be useful, but understanding how it works under the hood is really useful in my experience. It helps me to understand why design decisions were made in a certain way in abstractions, which lets me pick things up more quickly and ultimately become a better engineer.

If you'd like to take a look at the code yourself, it's [here](https://github.com/AndKaminer/ml-implementations). 

