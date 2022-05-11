# Tubable-Survivabilty-Off-chain
The repository contains the following:

1) PDF document with explanation of the work we done.
2) google colab notebook with a simple python implementation of our proposed algorithm.

## Intro
As block-chain usage increased in the last decade, its scalability restrictions are more relevant then ever.<br />
The fact that each transaction needs to be written inside a block, which is added to the chain on average time of 10 minutes, does not allow Block-chain based currency users to conduct quick transactions. Off-chain was introduced lately in order to allow block-chain users, to conduct those transactions with low latency waiting time.

The basic concept is creating a paying channel between 2 users, which is backed up with some currency on the original block-chain network.<br />
These deposits dictate the channel capacity, thus guarantee that no user would have unsettled debts.<br />

The created channels can be used as an external network:<br />
Users can transfer money on a path of several different existing channels, thus creating a new network, an "Off-chain" network.<br />
Since creating a new channel requires an "On-Chain" transaction, it is expensive and has low-latency. Therefore, we rather create an "Off-chain" topology that requires as least "On-chain" deposits as possible, while allowing as many "Off-chain" transactions as possible. In particular, we are interested in generating a topology that suffices the transaction requirements in a certain probability, which will be denoted as the network survivability.<br />

Our work focuses on those two restrictions in-order to give a good cost-effective solution.<br />
### Related Work
1. Previous work was done in finding optimal off-chain in terms of minimal deposits and transaction fees.<br />
2. The idea of graph survivability was initially presented as a constraint for routing trees.<br />

## Main Contribution
In our work we designed an Algorithm that helps maintain an Off-chain topology, s.t. The users payment demands holds and the On-chain deposits are minimized.

The algorithm have several assumptions:<br/>
1. Once a route path was chosen over the "Off-chain" network, all transaction between the users are done on the same path. if there are no sufficient deposit left, the transaction fails. <br/>
2. We are considering 0 fees with regards to the routing path chosen, thus making the deposit amount the only cost metric involved in our algorithm.  <br/>
3. We are assuming that the transaction demands are periodical and occur randomly with a given distribution over a specific time interval. 

## Experiments
We ran a simple simulation on our POC algorithm with the following configuration:<br /> 
    1. 10 different users (N=10). <br />
    2. Changing amount of off-chain payment demands. <br />
    3. We compared our algorithm results to a naive deposit per-demand approach, where we are establishing a payment channel for each demand between 2 users. <br />

### Results
* In blue - The naive method
* In Orange - Our method

![alt text][exp_result_1]
![alt text][exp_result_2]

<br />
The results were encouraging as when the number of total demands increased the
saving of our algorithm also increased.<br />
We noticed that at the first iterations the total deposits are equal, while
at a certain point, when there are enough channels that can be shared, the deposits in
our algorithm grows slower.

## Detailed Work
For more info on our suggested algorithm, read our PDF : [Tunable_Survivability_on_Off_chain_network.pdf](https://github.com/Ugadot/Tunable-Survivabilty-Off-chain/blob/main/Tunable_Survivability_on_Off_chain_network.pdf)

## Authors

Tunable Survivability Off Chain Algorithm was designed by [Uri Gadot](https://github.com/Ugadot) and [Hagay Michaeli](https://github.com/hmichaeli)



[exp_result_1]: https://github.com/Ugadot/Tunable-Survivabilty-Off-chain/blob/main/exp10(1).png "Logo Title Text 2"
[exp_result_2]: https://github.com/Ugadot/Tunable-Survivabilty-Off-chain/blob/main/exp11.png "Logo Title Text 2"
