# Polyrating

This repository contains a general-purpose library for generating ratings using well-known and new rating systems, such as Elo, Glicko, TrueSkill, and Polyrating. Specifically, it currently serves the following three main purposes:
- It is the official repository for the computation of the ETH Chess Ratings, based on the tournaments and events organized by the [ETH Chess Club](https://chess.ethz.ch/).
- It contains the code to reproduce the results from our paper *Polyrating: A Multivariate Rating System for Language Model Evaluation*. 
- It can be used to compute rating systems for people's projects.

This README is structured as follows:
- [Installation and Basic Use](#installation-and-basic-use): Provides details on how to install and use the code.
- [Further Use](#further-use): Provides a more in-depth look at various features of the code that can be useful for more advanced use cases. How to use our rating system, `Polyrating`, is also explained here in further detail.
- [Reproducing Results](#reproducing-results): Provides a step-by-step overview of how to reproduce the results from our paper.
- [ETH Chess Rating](#eth-chess-rating): Provides a detailed description of the method used to compute the chess ratings for the ETH Chess Club.

Feel free to open an issue when encountering any bug, having a feature request, or for any other questions regarding this repository.

## Installation and Basic Use

You can install the code by installing [Conda](https://docs.anaconda.com/free/miniconda/) and running the following in your command prompt:

```bash
cd path/to/this/folder
conda create -n ratings python=3.11 -y
conda activate ratings
python -m pip install -e .
```

This allows you to run the program.

### Basic Use

This repository provides an easy way to manage games, select a rating system, and obtain a leaderboard for the players in your database. For example, the following code sets up a manager and adds a few games to the database. Note that every rating system requires a rating period to be set, where a rating period is a period of time where ratings remain static.

```python
from rating import Manager, Elo, RatingPeriodEnum, DetailedLeaderboard
from datetime import timedelta, datetime

manager = Manager(
    rating_system=Elo(),
    rating_period_type=RatingPeriodEnum.TIMEDELTA, # sets the constant rating period to be defined by some timedelta
    custom_timedelta=timedelta(days=30), # use monthly rating periods
)

game = manager.add_game(
    "Jasper Dekoninck", # home player
    "Magnus Carlsen", # out player
    "1-0", # result, either 1-0, 0-1, 1/2-1/2
 datetime.now(), # data of the game
)
manager.add_game("Jasper Dekoninck", "Hikaru Nakamura", 
                 "1-0", datetime.now())
manager.add_game("Magnus Carlsen", "Hikaru Nakamura", 
                 "1/2-1/2", datetime.now())
manager.update_rating() # updates ratings
leaderboard = DetailedLeaderboard.compute(manager.player_database, manager.game_database) # computes leaderboard
manager.save('data/databases.json') # saves database
manager = Manager.load('data/databases.json') # load database
```

This basic script should be sufficient for most use cases. Note that you can use all of the following rating systems: `Glicko`, `Glicko2`, `TrueSkillThroughTime`, `EloPlusPLus`, `ChessMetrics` and of course, several variants of Polyrating: `PolyratingCrossEntropy`, `PolyRatingRao`, `PolyRatingDavidson`, and `PolyratingAccuracy`. The hyperparameters for each of these options are explained in the documentation. 

## Further Use
This section explains several further features implemented in this library.

### Tournaments
You can also manage tournaments within your rating system.

```python
from rating import Manager, Tournament
from datetime import datetime

manager = Manager()
tournament = Tournament("Our awesome tournament", datetime.now(), rounds=7, time_control='5+3') # rounds and time_control are completely optional
manager.add_tournament(tournament=tournament)
game = manager.add_game("Jasper Dekoninck", "Hikaru Nakamura", 
                 "1-0", datetime.now(), tournament_id=tournament.id) # adds the game to the tournament
```
By using the tournament object, one can compute tournament-specific statistics, as explained further in the section on [statistics](#statistics).

The ETH Chess Club uses the [VegaChess](https://www.vegachess.com/ns/) software to administer our tournaments. Therefore, we have implemented additional features that interact with this software for the automation of our pipeline.

```python
from rating import Manager, Tournament

manager = Manager()
manager.add_tournament("path/to/tournament", "Our awesome tournament") # automatically adds all games and players from the tournament that used VegaChess
```

### Default Rating
By default, player ratings are initialized at 1500 with a standard deviation of 500. To change this, simply do:
```python
from rating import DEFAULT_RATING
DEFAULT_RATING.set_default(rating=1000, deviation=250)
```

### Object Handling

Apart from adding games to the database, you can also invoke the following functions separately:
```python
from rating import Manager, Tournament
from datetime import datetime

manager = Manager()
tournament = Tournament("Our awesome tournament", datetime.now())
manager.add_tournament(tournament=tournament)
game = manager.add_game("Jasper Dekoninck", "Hikaru Nakamura", 
                 "1-0", datetime.now())
manager.add_player("Magnus Carlsen") # only adds a player
manager.remove_game(game=game)
manager.remove_player('Jasper Dekoninck') # removes the player and all games associated with that player
manager.remove_tournament(tournament=tournament) # removes the tournament and all games associated with the tournament
```

### Complex Results
It is possible to add more complex results to games. The following results are all valid:
- `1-0`, `0-1`, `1/2-1/2`: Standard results.
- `1-0F`, `1F-0`: Any `F` in a result will be counted as a forfeit. Forfeits are not used when computing ratings.
- `0.6-0.4`, `3-1`, `4-1`: More complex results. Some rating systems are able to leverage these results to obtain improved ratings.

Furthermore, a `Game` object can be instantiated with various point systems for computing the winner of a tournament. By default, each player gets 1 point for a win, 0.5 for a draw, and 0 for a loss. You can change these defaults as follows:
```python
from rating import Manager, Game
from datetime import datetime

manager = Manager()
player1 = manager.add_player('Manchester United')
player2 = manager.add_player('Manchester City')

game = Game(player1.id, player2.id, "2-2", datetime.now(), 
            points_for_win=3, points_for_tie=1, points_for_loss=0)
manager.add_game(game=game)
```

### Statistics
You can compute a variety of statistics on your data. The easiest way to compute these statistics is by simply calling `manager.compute_statistics()`. All statistics will then be automatically computed and stored in the `data` folder. In the following table, you can find the description of all statistics that you can compute.

| Statistic      | Description | Stored in |
| ----------- | ----------- | ----------- |
| Leaderboard | Contains the rating for each player. Removes players that have played 12 games or less. Used for the ETH Chess Club. | `leaderboard.csv` |
| DetailedLeaderboard | Contains the rating for each player. | `detailed_leaderboard.csv` |
| AnonymousLeaderboard | Contains the rating for each player. Player names are anonymized if they only played before May 2024. Used for the ETH Chess Club. | `anonymized_leaderboard.csv` |
| Tournament Ranking | Computes the ranking for a specific tournament, along with tournament performances. |`ranking.csv`|
| Win Rate by Color | Computes how often the home player wins/loses/draws. | `win_rate_by_color.png`|
| Rating Distribution | Computes a plot of the rating distribution over all players | `rating_distribution.png`|
| Tournaments per player | Shows a histogram of the amount of tournaments a player has played. | `tournaments_per_player.png`|
| Win Rating Difference | Shows a plot of the win probability over various rating differences | `win_rating_difference.png`|
| Games per Player | Shows the sum of the number of games per player. | `number_of_games.csv`|
| Most Surprising Games | Shows the games where a player beats the odds by beating (or drawing) a much stronger player. | `most_surprising_games.csv`|
| Most Surprising Performances | Shows the highest rating increases for one tournament, normalized by the deviation. | `most_surprising_performances.csv` |

### Polyrating and Multivariate rating
Finally, we explain how to use our own `Polyrating` system using this repository. To compute a multivariate rating, you first need to add metadata to each game and player. In this repository, we call this metadata `advantages`. For example, the following code shows how to add some basic advantages to a game:

```python
from rating import Manager, Game
from datetime import datetime

manager = Manager()
player1 = manager.add_player('Jasper Dekoninck')
player2 = manager.add_player('Hikaru Nakamura')

game = Game(player1.id, player2.id, "1-0", datetime.now(), 
            advantages_home={'home': 1, 'blitz': 1}, advantages_out={'home': 0, 'blitz': 1})
manager.add_game(game=game)
```
Only the rating systems based on `Polyrating` can use these advantages to fit a better rating. The following code shows a typical way to use advantages:
```python
from rating import Manager, Game, PolyratingCrossEntropy, Matching, DefaultRating

rating_system = PolyratingCrossEntropy(
    advantages={'blitz': DefaultRating(0, 50)}, # each player will have an additional blitz rating, initialized as 0 with deviation of 50
    shared_advantages=[('home', Matching(), DefaultRating(0, 50), 5)] # shared over all players
)
manager = Manager(rating_system=rating_system)
...
```
`Matching` is a class that enables you to match each shared advantage with a specific set of players. An empty matching means it matches all players (more than that you will likely never need). The 5 at the end indicates the extra deviation that is added between rating periods. More technically, it is the deviation between consecutive time steps of the Markov Chain for that rating. This ensures it does not need to stay constant over time, but can change a bit between rating periods.

This extra deviation between rating periods can also be added for the non-shared advantages as follows:
```python
rating_system = PolyratingCrossEntropy(
    advantages={'blitz': DefaultRating(0, 50)},
    omegas_advantages={'blitz': 5}, # extra deviation for the advantage
    omega=5, # extra deviation for the base ratings
    shared_advantages=[('home', Matching(), DefaultRating(0, 50), 5)]
)
```

Finally, the `linearized` parameter allows you to introduce an approximation in the rating system since it is quite expensive to run. Essentially, if you set this parameter to `k`, the rating system will only optimize the rating over the last `k` rating periods and use the rating of each player from before this period to initialize its ratings instead of the default rating. A bit complicated, but all that matters is that the lower you set this, the faster the algorithm works, but the more approximate it becomes.

## Reproducing Results

To reproduce the results, you first need to install the package as described above and then run:
```bash
python -m pip install -r requirements.txt
bash scripts/paper/main.sh
```
This will download all datasets and run the code. We note that each file in this bash script is run consecutively on a single core, which will take a lot of time. You can adjust the number of cores used in the file manually to run things quicker. Furthermore, each line in the script is annotated with the name of the figure or table that it generates the data for. Results are stored in the `results` folder in interpretable csv files. In the following list, we mention to which results in the paper the csv files correspond.

- Table 1a / Table 6a: `lmsys_released_shared_ratings.csv`
- Table 1b / Table 6b: `wildbench_released_shared_ratings.csv`
- Fig 2a: `sample_efficient_is_chinese.csv`
- Fig 2b: `sample_efficient_is_code.csv`
- Fig 2c: `sample_efficient_is_hard.csv`
- Fig 3a: `sample_efficient_wildbench.csv`
- Fig 3b: `sample_efficient_mixeval.csv`
- Fig 3c: `sample_efficient_is_code_is_chinese.csv`
- Table 2 / table 7: `leaderboard_polyrating.csv`
- Table 3 / Table 8: `leaderboard_univariate*.csv` (all files that start with `leaderboard_univariate`)
- Fig 4:  `alternatives.csv`


## ETH Chess Rating

In this section, we briefly introduce the concept of a rating system and explain the system we are using for the ratings computed for the ETH Chess Club.

### Universal Concepts of Rating Systems

There are a few concepts that are almost universal across all rating systems. The first is that your rating is not just a number $R \in \mathbb{R}$, but also has a standard deviation $S \in \mathbb{R}$ associated with it. This standard deviation is the uncertainty regarding your rating: the higher it is, the more uncertain the system is about your rating. In theoretical terms, it is usually assumed that your rating follows a normal distribution $\mathcal{N}(R, S)$.

Secondly, ratings are only updated after a certain rating period. This means that your rating is not changed every game, but only every couple of games. This increases the stability of the rating system. For our chess club, one rating period is the time between two consecutive tournaments.

### ELO and Glicko

While we do not use it, it is interesting to see the intuition behind two of the most common rating systems: ELO and Glicko. ELO essentially updates your rating based on how well you played against your opponents: if you won a lot of games against very strong opponents (with a high rating), you will gain a significant amount of points and you will gain less if those opponents were less strong. The opposite is also true: if you lose against weak opponents, you will lose a lot of rating points.

Glicko is an improved version of this system, where your deviation $S$ is also taken into account: if your deviation is high, you will make large rating jumps, since your rating is still very uncertain. If your deviation is low, rating humps will also be lower. 
### Problems with ELO and Glicko

However, both ELO and Glicko are problematic for datasets with lots of players that played very few games (like ours). A typical issue occurs when a very strong player plays their first games. People who lose against this strong player will lose a lot of rating points due to the low initial rating of the strong player. In our case, about $70\%$ of players only ever play one tournament and this would thus significantly bias results.

### Iterative Solving

Both ELO and Glicko only perform one update to the rating for each rating period. However, we can perform multiple updates instead. The strong player from before will have a low rating in the first update, but by iterating the process their rating will go up and the losses against this player will be counted less severely.

We do this iterative solving by computing the so-called maximum likelihood estimator of the system: for each game, we predict the probability of winning based on the current ratings and then try to maximize this probability. We can solve this maximization problem by using optimization algorithms, and thus ensure that our new players are rated and accounted for correctly.

### Polyrating

One final adjustment we make is that in the current update, we not only take into account games from this rating period but also from the previous ones. If a certain player did not play well in their first tournament, but was actually a very good player and played well in their second, we can retroactively update the rating of this player in the first tournament. This in turn leads to different updates for players who played this player in the first tournament, which can change results. Practically, we achieve this by telling our system that the rating of the same player for consecutive tournaments should be very close together.

### Tournament Performance
To determine your tournament performance, we pretend you have never played before the tournament and compute your rating as it would be after the tournament.

### Surprising Performances

To compute the most surprising performances and games, we compute which games and performances the rating system is the most surprised by, i.e. associates the lowest probability of happening with.

### Mathematical Details

Is the above explanation not good enough for you? Here are the full mathematical details of the system we use.

Let $ \{0, 1, ..., T\}$ be the space of all time periods. Each player $P$ gets a rating associated at time $t$, which we denote by a rating $R_t \in \mathbb{R}$. We assume that the rating is a Markov Chain, i.e. $R_{t_1} - R_{t_2} \sim \mathcal{N}(0, |t_1 - t_2|\omega^2)$ where $\omega \in \mathbb{R}$ is a hyperparameter of the system that indicates how quickly ratings can change between rating periods. Furthermore, we assume a prior on the ratings following a normal distribution, i.e. $\mathcal{N}(R_0, \sigma_0^2)$. The log probability of ratings based on this process (without taking into account the games!) is then:
$$\log p_{\text{prior}}(R_1, ..., R_n) = \sum_{i = 1}^n \log \phi\left(\frac{R_{i, 0} - R_0}{\sigma_0}\right) + \sum_{j=0}^{t-1} \log \phi\left(\frac{R_{i, j} - R_{i, j + 1}}{\omega}\right)$$
where $\phi$ is the probability density function of the standard normal distribution.

We then use the so-called Bradley-Terry Model to compute the expected value of the result of a game. Let $s \in [0,1]$ be the result of the game between $P_1$ and $P_2$ having taken place at time $t$. We denote such a game as $(s, R_1, R_2, t)$. The expected score is then:
$$\mathbb{E}[s] = \frac
{\gamma_1(t)}{\gamma_1(t) + \gamma_2(t)}$$
where
$$\gamma_i(t) = 10^{R_{i, t} / C}$$
Here, $C \in \mathbb{R}$ is a hyperparameter that indicates how different ratings have to be before they are significantly different, usually put as $C = 400$. We can now construct the formula for the log-likelihood of the ratings associated with the players based on all games $\{(s_i, R_{g_i}, R_{g_i'}, t_i) | i \in \{1, ..., m\}\}$:

$$\log p_{\text{games}}(P_1, ..., P_n) = \sum_{i = 1}^m s \log \frac
{\gamma_{g_i}(t)}{\gamma_{g_i}(t) + \gamma_{g_i'}(t)} + (1-s) \log \frac
{\gamma_{g_i'}(t)}{\gamma_{g_i}(t) + \gamma_{g_i'}(t)} $$

To derive this formula, we note that this is simply the binary cross-entropy of the Bradley-Terry model. Alternatively, you can count a tie as half a win and half a loss and note that:
$$\mathbb{P}(s_i = 1) = 1 - \mathbb{P}(s_i = 0) = \frac{\gamma_{g_i}(t)}{\gamma_{g_i}(t) + \gamma_{g_i'}(t)}$$
This will also get you to the above equation (the derivation is left as an exercise to the reader, as mathematicians like to say when they don't feel like writing out the derivation).

The full log probabilities now look like this:

$$\log p(P_1, ..., P_n) = \log p_{\text{games}}(P_1, ..., P_n) + \log p_{\text{prior}}(P_1, ..., P_n)$$

We can solve this set of equations using the Newton-Rhapson method. We note that this logprob is convex and that we additionally use Armijo's Rule to ensure that the solution does not diverge. We do not go into detail about these methods here, but you can look them up!

Finally, we have not explained yet how we obtain the standard deviation associated with the ratings. We make use of the Fisher information matrix for this purpose and note that for the maximum-likelihood estimator (which we compute), the following result holds (under some regularity conditions):
$$R_i \sim \mathcal{N}(\hat{R_{i}}, \mathcal{I}^{-1}(\hat{R_i}))$$
where $\hat{R_i}$ is the estimated rating, $R_{i}$ the actual (unknown) rating and $\mathcal{I}$ the Fisher Information Matrix. Using this equation, we can also estimate the most surprising events. We note that this deviation is just an approximation, and therefore not used in our paper.

## Citation

@article{dekoninck2024polyrating,
      title={Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation}, 
      author={Jasper Dekoninck and Maximilian Baader and Martin Vechev},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
