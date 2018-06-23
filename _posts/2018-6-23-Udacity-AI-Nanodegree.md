---
layout: post
title: Udacity AI Nanodegree
---

In this first review I will look at the following online course I've done recently (still need to finish the last project out of 4):  
[AI Nanodegree](https://www.udacity.com/course/ai-artificial-intelligence-nanodegree--nd898)

![_config.yml]({{ site.baseurl }}/images/keJie.jpg)
*Ke Jie Go champion 2017 defeated by AlphaGo*

With all the hype surrounding artificial intelligence I felt the need to go a bit further than machine learning. Maybe not as much as I'm focusing on ML but at least to get some degree of comprehension and knowledge. 
Neural Networks, traditional ML and many other methods aiming to model a stochastic problem are all over the news, in online courses, discussion forums etc. But it is self evident for anyone working on ML or studying it that this knowledge is certainly not able to build all things
that we consider artificial intelligence. The word "consider" here is important since the definition is always changing and is a source of debate around coffee machines: "Is matrix multiplication in a neural network really intelligence ?". 
That philosophical question is even addressed in one of the lecture. Anyway, I wanted to know more about the field in general rather than the ML part. What exactly is operations research ? How Google Maps work ? 
How to optimize schedules ? Production ? etc.

This is what this course mostly addresses and why I became interested. Here is what the course focuses on:
* **Constraint satisfaction problems**
	* Some problems have constraint to respect. That part of the program uses techniques called constraint propagation and domain knowledge to build an agent that can solve a Sudoku problem. It is mostly an intro to the rest of the course.
	* **Project** : Build a Sudoku solver
* **Search, Optimization and planning**
	* This part of the course focuses on tree searches algorithms (Breadth first, depth first, A*) and symbolic logic to build agents that try to achieve a defined goal state in the most efficient ways possible. 
	* **Project** : Build a forward planning agent
* **Adversarial search**
	* This time the goal is to beat another player. The course therefore focuses here on games where two players are involved and how an agent can search for the solution with the most potential for winning. It focuses mostly around the minimax algorithm 
	(and improvements like iterative deepening). The idea behind minimax is fairly simple. The agent must maximize its winning potential while taking into account that during adversary turns (when searching for a good move) the adversary will instead pick a move that minimizes our agent chances of winning.
	* **Project** : Build an agent that plays isolation game where both players have a chess piece and try to isolate the other player.
* **Fundamentals of Probabilistic Graphical models**
	* This part is making the bridge to another area I know better: machine learning. All previous agents always evolve in a fully observable deterministic universe however many problems are stochastic. It explores bayesian nets and hidden Markov models. 
	The subjects here (PGM) can probably be explored even further in [Daphne Koller course on coursera](https://www.coursera.org/learn/probabilistic-graphical-models). On the subject of Natural Language Processing though, I enjoyed the one from the [advanced Machine Learning](https://www.coursera.org/learn/language-processing/) better as it focuses on deep learning techniques and felt more up to date (and obviously covers more content). 
	* **Project** : Part of speech tagging. In a sentence, build a model that can recognize and put tags on each words.

**Who is the course for ?**  
The whole course uses Python programming. The exercises/projects throw code at you with not that much explanations and you will have to dig into the code given to understand what the different classes and methods do in order to be able to code the agents. There is no step by step instructions most of the time and you are given an empty function for an algorithm (and the pseudo code) and therefore must manage to construct the algorithm. 
It's a bit frustrating having to wrap your head around it first but as you poke the classes and fiddle with the code a bit, I always managed after a few hours to understand what I was supposed to be doing.  
You need a pretty solid base in Python (or at least another object oriented language). I found the time investment to be reasonable depending on how hard you find the problems I guess. Also a lot of the materials is optional readings so your time investment will vary depending on how much you want to get out of the course.
	
So now what did I think of the course ?  
**Pros:**
* I found the offer rather unique and good tools to add in my toolbox
* Demanding projects and exercises
* The community is excellent. The channels well organized for discussions and the instructors very present to answer questions. I also found a lot of tips and help just reading or speaking with other students. In comparison I usually find the Coursera forums rather empty and not very active.

**Cons:**
* Rather expensive for an online course
* The videos are not that great sometimes and some material is rushed. You need to compensate by reading.

The book used by this course is [Artificial Intelligence, a Modern Approach](https://en.wikipedia.org/wiki/Artificial_Intelligence%3A_A_Modern_Approach) by Russel and Norvig

