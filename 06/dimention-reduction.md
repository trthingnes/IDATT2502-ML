# Feature selection using Weka


## Task 1
**Using the UCI mushroom dataset from the last exercise, perform a feature selection using a classifier evaluator.
Which features are most discriminative?**

Using the ClassifierAttributeEval with Ranker I got the following result:

    Ranked attributes:
      0   22 habitat
      0    8 gill-size
      0    9 gill-color
      0    6 gill-attachment
      0    7 gill-spacing
      0    5 odor
      0   21 population
      0    2 cap-surface
      0    3 cap-color
      0    4 bruises?
      0   10 stalk-shape
      0   11 stalk-root
      0   12 stalk-surface-above-ring
      0   19 ring-type
      0   20 spore-print-color
      0   13 stalk-surface-below-ring
      0   18 ring-number
      0   17 veil-color
      0   16 veil-type
      0   15 stalk-color-below-ring
      0   14 stalk-color-above-ring
      0    1 cap-shape

It seems like the habitat feature is one of the better ones at discriminating edibility,
closely followed by gill size, color, attachment, and spacing.


## Task 2
**Use principal components analysis to construct a reduced space.
Which combination of features explain the most variance in the dataset?**

    Ranked attributes:
     1   110 habitat=WASTE
     1    35 gill-color=PINK
     1    37 gill-color=GRAY
     1    38 gill-color=BLACK
     1    39 gill-color=CHOCOLATE
     1    36 gill-color=BROWN
     1    34 gill-color=WHITE
     1    41 gill-color=GREEN
     1    33 gill-size=BROAD
     1    30 odor=MUSTY
     1    31 gill-attachment=ATTACHED
     1    32 gill-spacing=CLOSE
     1    40 gill-color=PURPLE
     1    42 gill-color=RED
     1    28 odor=FISHY
     1    49 stalk-root=ROOTED
     1    51 stalk-surface-above-ring=SMOOTH
     1    52 stalk-surface-above-ring=FIBROUS
     1    53 stalk-surface-above-ring=SILKY
     1    50 stalk-root=EQUAL
     1    48 stalk-root=CLUB
     1    43 gill-color=BUFF
     1    47 stalk-root=BULBOUS
     1    44 gill-color=YELLOW
     1    45 gill-color=ORANGE
     1    46 stalk-shape=ENLARGING
     1    29 odor=SPICY
     1    27 odor=FOUL
     1   109 habitat=LEAVES
    ...

    Ranked attributes:
    0.9112     1 -0.256stalk-surface-above-ring=SILKY
                 -0.256stalk-surface-below-ring=SILKY
                 -0.231ring-type=LARGE
                 -0.23odor=FOUL
                 -0.204spore-print-color=CHOCOLATE
                 +0.196stalk-surface-above-ring=SMOOTH
                 +0.193ring-type=PENDANT
                 -0.193stalk-root=BULBOUS
                 +0.187stalk-color-below-ring=WHITE
                 +0.185stalk-color-above-ring=WHITE
                 +0.181stalk-surface-below-ring=SMOOTH
                 +0.179odor=NONE
                 -0.166bruises?=NO
                 +0.146stalk-root=EQUAL
                 +0.141spore-print-color=BROWN
                 -0.139population=SEVERAL
                 -0.138gill-color=BUFF
                 +0.136spore-print-color=BLACK
                 -0.136habitat=PATHS
                 +0.133population=SCATTERED
                 -0.131gill-spacing=CLOSE
                 -0.129stalk-color-below-ring=BUFF
                 -0.129stalk-color-above-ring=BUFF
                 -0.127stalk-color-above-ring=BROWN
                 -0.125stalk-color-below-ring=PINK ...
    ...
 

## Task 3
**Do you see any overlap between the PCA features and those obtained from feature selection?**

Both the most discriminative features and the features preferred by PCA seems to be the same.
The same features top both lists. This should make sense given that we are keeping the most significant
features while reducing the rest.
