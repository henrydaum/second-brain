**What Factorio Taught Me About Building Intelligent Systems**

*Belts, Trains, Robots, and the Architecture of Agentic AI*

Henry Daum

I recently played Factorio for more than forty hours in a single week. If you’re not familiar, Factorio is a game about building automated factories on an alien planet. You mine resources, assemble components, and construct increasingly elaborate systems to produce increasingly complex outputs. It’s the kind of game that makes you forget to eat.

But somewhere around hour thirty, I stopped seeing a game and started seeing an architecture. Factorio doesn’t just simulate logistics—it forces you to solve logistics, from first principles, at scale. And the solutions it converges on are strikingly similar to patterns I’ve encountered in distributed systems, data engineering, and AI. The game essentially asks: how do you move the right things to the right places at the right times, reliably, efficiently, and flexibly? That’s not just a game design question. It’s the central question of any large-scale agentic system.

Factorio answers this question with exactly three transport systems: belts, trains, and flying robots. Each operates at a different level of intelligence, speed, precision, and scale. None of them alone is sufficient. Together, they form something greater than the sum of their parts—a layered architecture where simplicity at the bottom supports complexity at the top. I want to unpack how each system works, what engineering principles it embodies, and what it suggests about building intelligent, autonomous systems in the real world.

**System 1: Belts — The Reliable Pipeline**

Transport belts are the simplest system in Factorio, and they’re also the most important. A belt is a tile you place on the ground. It has a direction—up, down, left, or right—and it moves items in that direction at a constant rate. When one belt faces into another, the items flow from one to the next. Chain enough belts together and you have a pipeline.

That’s it. No configuration screens, no programming, no conditions. Belts are stateless and deterministic. They move items at a fixed throughput—fifteen items per second for the basic tier—and they never surprise you. If you set up a line of belts from Point A to Point B, items will flow from A to B at a predictable rate, forever. This predictability is their superpower.

Belts also exhibit emergent behavior that mirrors real-world systems in interesting ways. Each belt tile acts as a two-lane, one-way road, and items on belts behave like traffic. When the front of the line stalls—say, because a machine downstream is full—items back up behind it, creating a natural buffer. When the blockage clears, the whole line surges forward. This is exactly how backpressure works in streaming architectures like Kafka or RabbitMQ: producers slow down when consumers can’t keep up, and the queue absorbs the difference.

Splitters add a layer of routing. A splitter takes one or two input belts and distributes items evenly across one or two output belts. If one output backs up, all items divert to the other—a crude but effective form of load balancing. Splitters can also filter by item type, acting as routers that sort mixed streams into dedicated lanes. Underground belts let lines cross each other without interference, functioning like tunnels or virtual channels in a network.

The key insight about belts is that their simplicity is a feature, not a limitation. Belts are easy to reason about, easy to debug, and easy to scale horizontally. When something goes wrong in a belt-based setup, you can literally see the problem: items are backed up here, missing there, going to the wrong place over there. Compare that to debugging a distributed message queue in production. The principle is the same: for your foundational data movement layer, you want something dead simple, visually inspectable, and rock-solid. Complexity belongs higher in the stack.

**System 2: Trains — The Batch Processor**

Belts work beautifully for short distances and steady flows. But Factorio maps are enormous, and as your factory grows, you need to move large volumes of material across long distances. Running a belt line for two thousand tiles is technically possible but wildly impractical—slow, expensive, and a nightmare to maintain. This is where trains come in.

A train in Factorio consists of one or more locomotives pulling cargo wagons, running on a dedicated rail network. Cargo wagons hold vast quantities of items. The train is programmed with a schedule: go to Station A, wait until cargo is full, then go to Station B, wait until cargo is empty, repeat. Trains are fast—far faster than belts—and they carry far more per trip.

If belts are streaming pipelines, trains are batch jobs. They don’t provide continuous flow; they provide high-throughput, high-latency transfers. You accumulate a big load, ship it all at once, then accumulate again. This is the same pattern behind ETL pipelines, nightly data syncs, and batch model training runs. Sometimes you don’t need real-time delivery. Sometimes it’s more efficient to wait, aggregate, and move in bulk.

But trains introduce coordination problems that belts never face. Multiple trains on the same rail network can collide and deadlock. A collision—one train plowing into another at an intersection—is catastrophic. Deadlock—two trains each blocking the other’s path—is subtler but equally fatal to throughput. Factorio solves this with rail signals, which divide the track into segments. A signal turns red when its segment is occupied, preventing other trains from entering. This is, almost exactly, a mutex or semaphore from concurrent programming. The rail network becomes a shared resource, and signals are the locks that prevent race conditions.

Trains also use the A\* pathfinding algorithm to navigate from station to station. When a segment is occupied, the pathfinder routes around it if possible, dynamically adapting to network conditions. This is analogous to adaptive routing in computer networks, where packets find alternate paths when links are congested. The train system, in other words, is a miniature distributed network with real-time path optimization and resource contention management.

The tradeoff is clear: trains give you throughput and speed at the cost of infrastructure complexity. Rail networks require careful layout, signal placement, and intersection design. They’re not something you throw together casually. But for the problems they solve—bulk transport, long distance, high volume—nothing else comes close.

**System 3: Robots — The Intelligent Agent**

Flying robots are the most sophisticated transport system in Factorio, and they operate on fundamentally different principles than belts or trains. Where belts are fixed infrastructure and trains follow predetermined routes, robots are autonomous agents that can go anywhere within a defined network and make decisions about what to carry and where to carry it.

The system centers on roboports—structures that house robots and define a coverage area. When roboports are placed close enough that their ranges overlap, their networks merge into a single, unified logistic network. Within this network, every item in every connected chest is part of one shared inventory. Requester chests can call for specific items, provider chests offer items up for transport, and storage chests absorb overflow. Robots autonomously pick up items from providers and deliver them to requesters, choosing the shortest paths and recruiting the nearest available robots for each job.

This is, functionally, a distributed task allocation system. The logistic network maintains a global state—a registry of all available items and all pending requests. When a new request comes in, the system identifies the nearest robots to the item source, dispatches them, and handles edge cases like the request being fulfilled by another source before the robots arrive. It’s reactive, parallel, and self-organizing. If you squint, it looks a lot like a Kubernetes scheduler, matching workloads to available resources based on proximity and capacity.

But the real magic is in the construction robots. Logistic robots move items. Construction robots use them. Given a blueprint—a template specifying which structures go where—construction robots will autonomously fetch the required materials from the network and physically place each structure, tile by tile. They can also deconstruct, clearing out areas by removing structures and returning materials to storage. The player issues high-level intent (“build this factory layout here”), and the robots handle all the execution details.

This is agentic behavior in the truest sense. Construction robots don’t just transport—they act on the world. They interpret a plan, source materials, and execute physical changes to the environment. They can even extend their own network by placing new roboports, expanding their operational range autonomously. The system is self-modifying: it can grow and restructure itself based on the player’s high-level instructions.

The tradeoff with robots is unpredictability. Because robots choose paths dynamically and their travel times depend on distance and availability, the exact timing of any given operation is hard to predict. A belt moves items at fifteen per second, always. A train arrives on a fixed schedule. But robots? They’ll get there when they get there. For a system that needs precise, metronomic timing, robots are the wrong tool. For a system that needs flexibility, precision of placement, and the ability to respond to novel situations, robots are unmatched.

**The Intelligence Gradient**

One of the most elegant aspects of Factorio’s design is the intelligence gradient across these three systems. Belts are essentially mechanical—no logic, no decisions, just physics moving items along a path. Trains are semi-intelligent: they compute optimal routes, respond to network conditions, and follow conditional schedules. Robots are fully autonomous agents with global awareness, dynamic task allocation, and the ability to modify the physical environment.

This gradient isn’t decorative. It’s architecturally necessary. Each level of intelligence comes with costs: complexity, unpredictability, resource consumption, and failure modes. Belts never crash, never deadlock, never run out of power. Trains can deadlock if signals are misconfigured. Robots can drain the power grid, swarm inefficiently, and exhibit hard-to-debug timing issues. The game teaches you, through painful experience, to use the simplest system that solves the problem and to reach for intelligence only when simplicity isn’t enough.

This is a principle that software engineers rediscover constantly. Don’t use a microservice when a function call will do. Don’t use a database when an in-memory cache suffices. Don’t deploy an ML model when a regex matches. Factorio’s three-system architecture is a visceral lesson in this: reliability at the base, intelligence at the top, and clear interfaces between the layers.

**The Emergent Whole**

In a mature Factorio base, all three systems work together in a layered composition. Trains haul bulk resources from distant mining outposts to a central hub. At the hub, belts distribute those resources to nearby assemblers in tight, predictable flows. Robots handle the last-mile delivery of specialized components, manage inventory across the base, and execute construction projects that expand the factory.

Each system covers for the others’ weaknesses. Trains can’t do precision work, but they feed the belts that can. Belts can’t span long distances efficiently, but trains bridge the gap. Neither trains nor belts can adapt to ad-hoc requests or build new infrastructure, but robots can. The whole is not just greater than the sum of its parts—it’s qualitatively different. It’s a system that can transport, process, adapt, and grow.

This compositional design has a name in systems engineering: layered architecture. Each layer has a well-defined role, communicates with adjacent layers through clear interfaces, and can be reasoned about independently. The internet works this way—physical layer, data link, network, transport, application. Operating systems work this way—hardware abstraction, kernel, system calls, user space. Factorio, whether intentionally or not, implements the same pattern for logistics.

**Toward an Agentic AI Architecture**

So what does any of this have to do with AI?

I’ve been thinking a lot about how to build AI systems that don’t just respond to prompts but actively accomplish goals in complex environments. The term for this is “agentic AI,” and it’s one of the most active frontiers in the field. The challenge isn’t making a model that’s smart—we have that. The challenge is making a system that’s smart, reliable, scalable, and able to act on the world. That’s a systems engineering problem, and Factorio offers a surprisingly coherent blueprint.

Here’s how the mapping works:

The belt layer corresponds to reliable, stateless data pipelines. In an AI system, this is the infrastructure that moves data from sources to processors to storage: message queues, streaming platforms, API gateways, and ETL scripts. The belt layer doesn’t think. It just moves data from A to B, predictably and tirelessly. It’s Kafka, it’s Redis streams, it’s a well-designed REST API. You want this layer to be boring. Boring is reliable.

The train layer corresponds to batch processing and heavy computation. This is where your model training runs live, your large-scale data transformations, your periodic knowledge base updates. Like trains, these operations are high-throughput but high-latency. You don’t retrain a model in real-time; you accumulate data, schedule a run, and ship the results. The coordination challenges are analogous too: you need to manage resource contention (GPU scheduling), avoid deadlocks (circular dependencies in pipelines), and route around failures (checkpoint and retry). The train layer is powerful but needs careful orchestration.

The robot layer corresponds to autonomous agents—LLM-powered systems that can interpret goals, plan actions, use tools, and modify their environment. Like Factorio’s robots, these agents operate within a network (the tools and APIs available to them), maintain awareness of shared state (memory, knowledge graphs, context), and can be dispatched to handle tasks dynamically. A logistic robot fetching items from a provider chest is an LLM agent calling an API to retrieve data. A construction robot placing a blueprint is an LLM agent executing a multi-step plan to build something—generating code, creating files, deploying services.

The parallel between construction robots and tool-using AI agents is especially striking. Construction robots don’t just move materials—they transform the environment according to a plan. They’re the only system in Factorio that can create new infrastructure, including expanding the network they operate within. Tool-using LLM agents have the same capability: they can write code that creates new tools, build workflows that extend their own reach, and recursively improve the systems they inhabit. This self-modification capacity is what makes robots—and agents—qualitatively different from pipelines and batch jobs.

And then there’s the fourth system, one that Factorio only gestures at: the orchestrator. In the game, this role is filled by the player. The player decides the high-level strategy: where to expand, what to produce, which bottlenecks to address. The player issues the blueprints that construction robots execute and designs the rail networks that trains traverse. The player is the intelligence that ties the three systems into a coherent whole.

In an agentic AI architecture, this orchestrator would be an AI system itself—a meta-agent that monitors the overall state of the system, identifies goals, allocates resources, and delegates execution to the appropriate layer. Simple, repetitive data movement? Route it through the pipeline layer. Large-scale computation? Schedule a batch job. Novel, complex task requiring judgment and tool use? Dispatch an agent. The orchestrator doesn’t do the work—it decides what work needs doing and which system should do it.

This isn’t purely theoretical. Modern AI systems are already groping toward this architecture. LangChain and similar frameworks let you compose agents with tools and memory. Kubernetes orchestrates containerized workloads across clusters. Apache Airflow schedules and monitors complex data pipelines. But these are typically separate systems, built by separate teams, with separate mental models. What Factorio suggests is that they should be designed as a unified whole—three complementary layers, each with a well-defined role, composed into a system that can transport, process, adapt, and grow.

**The Deeper Lesson**

The biggest takeaway from Factorio isn’t any single system—it’s the design philosophy that produced all three. The game demonstrates that large-scale automation doesn’t emerge from one brilliant system but from the disciplined composition of simple, specialized systems. Belts aren’t impressive individually. Neither are trains or robots. But together, they produce factories of extraordinary complexity and efficiency.

The same is true for AI. The field’s current obsession with making individual models more capable is important, but it’s only one dimension. The harder—and arguably more important—problem is systems integration: how do you combine a reliable data layer, a powerful computation layer, and a flexible agent layer into something that works as a unified whole? How do you build clear interfaces between these layers so that each can evolve independently? How do you ensure that the simplest layer handles the most volume, while intelligence is reserved for the tasks that actually require it?

Factorio doesn’t answer all of these questions. It’s a two-dimensional, tile-based game, and real-world systems are messier by orders of magnitude. But as a mental model—a way of thinking about layered, composable, scalable automation—it’s remarkably useful. Forty hours in a week is a lot of Factorio. But the systems thinking it instills is worth every minute.