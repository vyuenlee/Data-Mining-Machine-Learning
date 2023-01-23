#------------------------------------------------------------------------------
#
# Data Mining & Machine Learning
# Title: Bath Soap Market Segmentation
# By: Vivian Yuen-Lee
#
#------------------------------------------------------------------------------


# Load all necessary libraries 
library(ggplot2)          # library for sophisticated plots
library(dplyr)            # library for creating profile summaries
library(caret)            # library for stratified random split
library(rpart)            # library for decision tree model
library(rpart.plot)       # library for plotting a decision tree

# Load the csv file
soap_df <- read.csv("BathSoapHousehold.csv")
str(soap_df)                        # take a look at the data structure
# There are 600 observations of 46 variables
View(soap_df)
head(soap_df, 10)                   # examine the first 10 rows


#------------------------------------------------------------------------------
# Use k-means clustering to identify clusters of households based on purchase
# behavior, basis for purchase, and both parameters

#------------------------------------------------------------------------------
# Evaluating based on purchase behavior including brand loyalty
# Brand loyalty can be measured from 3 different perspectives:
# i)	The number of different brands purchase by the customer
# ii)	How often the customer switch from one brand to another (including 
# susceptibility to discounts/promotions), and
# iii)	Percentage of purchases spent on different brands
#
# Variables that describe purchase behavior include: 
# $ No..of.Brands        : int  3 5 5 2 3 3 4 3 2 4 ...
# $ Brand.Runs           : int  17 25 37 4 6 26 17 8 12 13 ...
# $ Total.Volume         : int  8025 13975 23100 1500 8300 18175 9950 9300 26490
# $ No..of..Trans        : int  24 40 63 4 13 41 26 25 27 18 ...
# $ Value                : num  818 1682 1950 114 591 ...
# $ Avg..Price           : num  10.19 12.03 8.44 7.6 7.12 ...
# $ Pur.Vol.No.Promo.... : num  1 0.887 0.942 1 0.614 ...
# $ Pur.Vol.Promo.6..    : num  0 0.0966 0.0195 0 0.1446 ...
# $ Pur.Vol.Other.Promo..: num  0 0.0161 0.039 0 0.241 ...
# $ Br..Cd..57..144      : num  0.3769 0.0215 0.026 0.4 0.0482 ...
# $ Br..Cd..55           : num  0.1308 0.0751 0.5455 0.6 0.1446 ...
# $ Br..Cd..272          : num  0 0 0 0 0 ...
# $ Br..Cd..286          : num  0 0 0.0303 0 0 ...
# $ Br..Cd..24           : num  0 0 0 0 0 0 0 0 0 0 ...
# $ Br..Cd..481          : num  0 0.059 0 0 0 ...
# $ Br..Cd..352          : num  0 0 0 0 0 0 0 0 0 0 ...
# $ Br..Cd..5            : num  0 0.1449 0.0195 0 0 ...
# $ Others.999           : num  0.492 0.699 0.379 0 0.807 ...

# The last 9 variables are proportions of purchases that were spent on any of 
# these specific brands.
soap_df[1:10,23:31]         # display the first 10 records

# However, including the percentages of total purchases for various brands is
# inefficient, may contain redundant information, and increase between-record / 
# within-cluster distances.
# Note that a customer who buys all brand A is as brand-loyal as someone who  
# buys all brand B.  In order to derive a single variable that represents brand 
# loyalty based on these percentages, we can consolidate the 8 brand percentages 
# by finding the maximum value for each record.  If a customer shows a high 
# percentage/proportion in any of the brands, then he/she would be interpreted 
# as a loyal customer.
# "Others" will remain as a separate variable since it may consist of a number 
# of other different brands.  A high percentage in this variable is an 
# indication that the customer is not brand-loyal.

# Add a derived variable that is the parallel maxima of the 8 brands
soap_df$MaxBrandPur <- pmax(soap_df$Br..Cd..57..144, soap_df$Br..Cd..55,
                            soap_df$Br..Cd..272, soap_df$Br..Cd..286,
                            soap_df$Br..Cd..24, soap_df$Br..Cd..481,
                            soap_df$Br..Cd..352, soap_df$Br..Cd..5)

# Copy these variables to a new data frame
behavior_df <- soap_df[ , c(12:16, 19:22, 31, 47)]
str(behavior_df)                # make sure all selected variables are there

# Compute normalized distance
behavior_norm <- sapply(behavior_df, scale)
row.names(behavior_norm) <- row.names(behavior_df)

# Examine the different numbers of clusters (between 2 and 5)
set.seed(123)                 # set seed to make kmeans() reproducible

# Compare with other k values between 2 and 5:
# Run the k-means algorithm for k=2
behavior_km2 <- kmeans(behavior_norm, 2)
# Show size of clusters
behavior_km2$size
# There are 2 relatively evenly-sized clusters

# Run the k-means algorithm for k=3
behavior_km3 <- kmeans(behavior_norm, 3)
# Show size of clusters
behavior_km3$size
# There are 3 evenly-sized clusters 

# Run the k-means algorithm for k=4
behavior_km4 <- kmeans(behavior_norm, 4)
# Show size of clusters
behavior_km4$size
# There are 3 evenly-sized clusters and 1 small cluster

# Run the k-means algorithm for k=5
behavior_km5 <- kmeans(behavior_norm, 5)
# Show size of clusters
behavior_km5$size
# There are 3 evenly-sized clusters  and 2 smaller clusters

# Check the within-cluster sum of squared distances for each k:
# k = 2
behavior_km2$withinss
# Compute the within-cluster average sum of squared distances
mean(behavior_km2$withinss)
# k = 3
behavior_km3$withinss
# Compute the within-cluster average sum of squared distances
mean(behavior_km3$withinss)
# k = 4
behavior_km4$withinss
# Compute the within-cluster average sum of squared distances
mean(behavior_km4$withinss)
# k = 5
behavior_km5$withinss
# Compute the within-cluster average sum of squared distances
mean(behavior_km5$withinss)

# Use the graphical method to choose the best k: 
# k should be chosen such that the between-cluster distance are highest while 
# the within-cluster sum of squared distances are smallest.

# Plot the average within-cluster sum of squared distances
avg_dist <- vector()
for (k in 1:10){
  km <- kmeans(behavior_norm, k)
  avg_dist[k] <- mean(km$withinss)
}
x <- 1:10
plot(x,avg_dist, type = "l",
     xlab = "# of clusters", ylab = "Avg Within-Cluster Distance",
     main = "K vs. Within-Cluster Distance (Purchase Behavior)")

# Moving from k=1 to k=2 yields a significant reduction in within-cluster 
# distance.  Notable improvements can also be seen by changing from 2 clusters 
# to 3 clusters, and from 3 clusters to 4 clusters.  There is a slight 
# improvement by changing from 4 clusters to 5 clusters and from 5 clusters to 
# 6 clusters.  Adding more clusters beyond 6 brings less improvement to cluster 
# homogeneity.  However, since market effort can only support up to 5 marketing 
# approaches, k=5 seems to be the most optimal choice.

# Compute the distance between cluster centroids
dist(behavior_km2$centers)
dist(behavior_km3$centers)
dist(behavior_km4$centers)
dist(behavior_km5$centers)
# None of the clusters appear unreasonable or are outliers

# Generate a plot to characterize the resulting clusters:
# Generate profile plot of centroids for k=2
# First, make an empty scatter plot
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(behavior_km2$centers), max(behavior_km2$centers)), 
     xlim = c(0,11), 
     main = "Profile Plot of Purchase Behavior Cluster Centroids (k=2)")
# Label x-axis
axis(1, at = c(1:11), labels = names(behavior_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:2))         # k=2
  lines(behavior_km2$centers[i, ], lty = i, lwd = 2, 
        col = "steel blue")
# Add cluster names
text(x = 0.5, y = behavior_km2$centers[ ,1], 
     labels = paste("Cluster ", c(1:2)), cex = 0.8)

# Generate profile plot of centroids for k=3
# First, make an empty scatter plot
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(behavior_km3$centers), max(behavior_km3$centers)), 
     xlim = c(0,11), 
     main = "Profile Plot of Purchase Behavior Cluster Centroids (k=3)")
# Label x-axis
axis(1, at = c(1:11), labels = names(behavior_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:3))         # k=3
  lines(behavior_km3$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,3),"blue", "sky blue"))
# Add cluster names
text(x = 0.5, y = behavior_km3$centers[ ,1], 
     labels = paste("Cluster ", c(1:3)), cex = 0.8)

# Generate profile plot of centroids for k=4
# First, make an empty scatter plot
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(behavior_km4$centers), max(behavior_km4$centers)), 
     xlim = c(0,11), 
     main = "Profile Plot of Purchase Behavior Cluster Centroids (k=4)")
# Label x-axis
axis(1, at = c(1:11), labels = names(behavior_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:4))         # k=4
  lines(behavior_km4$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,4),"sky blue", "blue"))
# Add cluster names
text(x = 0.5, y = behavior_km4$centers[ ,1], 
     labels = paste("Cluster ", c(1:4)), cex = 0.8)

# Generate profile plot of centroids for k=5
# First, make an empty scatter plot
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(behavior_km5$centers), max(behavior_km5$centers)), 
     xlim = c(0,11), 
     main = "Profile Plot of Purchase Behavior Cluster Centroids (k=5)")
# Label x-axis
axis(1, at = c(1:11), labels = names(behavior_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:5))         # k=5
  lines(behavior_km5$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,3, 5),"sky blue", "blue"))
# Add cluster names
text(x = 0.5, y = behavior_km5$centers[ ,1], 
     labels = paste("Cluster ", c(1:5)), cex = 0.8)

# Refer to written report for interpretation of these plots.
# k=5 is the recommended choice.

#------------------------------------------------------------------------------
# Evaluating based on the variables that describe the basis for purchase
# Variables that describe the basis for purchase include: 
# $ Avg..Price           : num  10.19 12.03 8.44 7.6 7.12 ...
# $ Pur.Vol.No.Promo.... : num  1 0.887 0.942 1 0.614 ...
# $ Pur.Vol.Promo.6..    : num  0 0.0966 0.0195 0 0.1446 ...
# $ Pur.Vol.Other.Promo..: num  0 0.0161 0.039 0 0.241 ...
# $ Pr.Cat.1             : num  0.234 0.293 0.12 0 0 ...
# $ Pr.Cat.2             : num  0.5607 0.5474 0.3182 0.4 0.0482 ...
# $ Pr.Cat.3             : num  0.1308 0.0948 0.5617 0.6 0.1446 ...
# $ Pr.Cat.4             : num  0.0748 0.0644 0 0 0.8072 ...
# $ PropCat.5            : num  0.502 0.456 0.245 0.4 0.807 ...
# $ PropCat.6            : num  0 0.347 0.121 0 0 ...
# $ PropCat.7            : num  0 0.0268 0.0335 0 0 ...
# $ PropCat.8            : num  0 0.0161 0.0108 0 0.0482 ...
# $ PropCat.9            : num  0 0.01431 0.00866 0 0 ...
# $ PropCat.10           : num  0 0 0 0 0 0 0 0 0 0 ...
# $ PropCat.11           : num  0 0.059 0 0 0 ...
# $ PropCat.12           : num  0.028 0 0.0162 0 0 ...
# $ PropCat.13           : num  0 0 0 0 0 0 0 0 0 0 ...
# $ PropCat.14           : num  0.1308 0.0805 0.5617 0.6 0.1446 ...
# $ PropCat.15           : num  0.33956 0 0.00325 0 0 ...

# The last 11 variables are percentages of purchases that were made under the 
# specific product proposition category.
soap_df[1:10,36:46]         # display the first 10 records

# Add a derived variable that is the parallel maxima of the 11 propositions
soap_df$MaxProp <- pmax(soap_df$PropCat.5, soap_df$PropCat.6, 
                        soap_df$PropCat.7, soap_df$PropCat.8, 
                        soap_df$PropCat.9, soap_df$PropCat.10,
                        soap_df$PropCat.11, soap_df$PropCat.12,
                        soap_df$PropCat.13, soap_df$PropCat.14,
                        soap_df$PropCat.15)

# Add a second derived variable to indicate which product proposition category
# is the highest
# For each row i, determine which of the proposition category is the max and 
# store this category in the new variable
soap_df$MaxPropCat <- NA
for (i in 1:dim(soap_df)[1]){
  for(j in 36:46){
    if(soap_df[i,48] == soap_df[i,j])
      soap_df[i,49] = j - 31      # Proposition categories are between 5 & 15
  }
}

# Copy these variables to a new data frame
basis_df <- soap_df[ , c(19:22, 32:35, 48:49)]
str(basis_df)               # check to make sure all selected columns are there

# Compute normalized distance
basis_norm <- sapply(basis_df, scale)
row.names(basis_norm) <- row.names(basis_df)

# Examine the different numbers of clusters (between 2 and 5)
set.seed(1234)                 # set seed to make kmeans() reproducible
# Run the k-means algorithm for k=2
basis_km2 <- kmeans(basis_norm, 2)
# Show size of clusters
basis_km2$size
# There is 1 larger cluster and 1 slightly smaller cluster

# Run the k-means algorithm for k=3
basis_km3 <- kmeans(basis_norm, 3)
# Show size of clusters
basis_km3$size
# There is 1 large cluster and 2 small clusters

# Run the k-means algorithm for k=4
basis_km4 <- kmeans(basis_norm, 4)
# Show size of clusters
basis_km4$size
# There is 1 large cluster, 1 medium cluster and 2 smaller clusters

# Run the k-means algorithm for k=5
basis_km5 <- kmeans(basis_norm, 5)
# Show size of clusters
basis_km5$size
# There is 1 larger cluster, 1 medium-sized cluster and 3 smaller clusters

# Check the within-cluster sum of squared distances for each k:
# k = 2
basis_km2$withinss
# Compute the within-cluster average sum of squared distances
mean(basis_km2$withinss)
# k = 3
basis_km3$withinss
# Compute the within-cluster average sum of squared distances
mean(basis_km3$withinss)
# k = 4
basis_km4$withinss
# Compute the within-cluster average sum of squared distances
mean(basis_km4$withinss)
# k = 5
basis_km5$withinss
# Compute the within-cluster average sum of squared distances
mean(basis_km5$withinss)

# # Use the graphical method to choose the best k:  
# k should be chosen such that the between-cluster distance are highest while 
# the within-cluster sum of squared distances are smallest.

# Plot the average within-cluster sum of squared distances
avg_dist_b <- vector()
for (k in 1:10){
  km_b <- kmeans(basis_norm, k)
  avg_dist_b[k] <- mean(km_b$withinss)
}
x <- 1:10
plot(x,avg_dist_b, type = "l",
     xlab = "# of clusters", ylab = "Avg Within-Cluster Distance",
     main = "K vs. Within-Cluster Distance (Basis for Purchase)")

# Moving from k=1 to k=2 and from k=2 to k=3 yield significant reductions in 
# within-cluster distances.  Improvements can be achieved by changing from 3 to 
# 4 and from 3 to 4 clusters.  From 4 to 6 clusters start to show less 
# improvement to homogeneity.  Similar to the previous analysis, improvements 
# beyond 6 clusters is very subtle.

# Compute the distance between cluster centroids
dist(basis_km2$centers)
dist(basis_km3$centers)
dist(basis_km4$centers)
dist(basis_km5$centers)
# All intra-cluster distances appear reasonable with no obvious outliers 
# detected.

# Generate a plot to characterize the resulting clusters
# Generate profile plot of centroids for k=2
# First, make an empty scatter plot
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(basis_km2$centers), max(basis_km2$centers)), xlim = c(0,10), 
     main = "Profile Plot of Basis for Purchase Cluster Centroids (k=2)")
# Label x-axis
axis(1, at = c(1:10), labels = names(basis_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:2))         # k=2
  lines(basis_km2$centers[i, ], lty = i, lwd = 2, 
        col = "blue")
# Add cluster names
text(x = 0.5, y = basis_km2$centers[ ,1], 
     labels = paste("Cluster ", c(1:2)), cex = 0.8)

# Generate profile plot of centroids for k=3
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(basis_km3$centers), max(basis_km3$centers)), xlim = c(0,10), 
     main = "Profile Plot of Basis for Purchase Cluster Centroids (k=3)")
# Label x-axis
axis(1, at = c(1:10), labels = names(basis_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:3))         # k=3
  lines(basis_km3$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,3),"blue", "sky blue"))
# Add cluster names
text(x = 0.5, y = basis_km3$centers[ ,1], 
     labels = paste("Cluster ", c(1:3)), cex = 0.8)

# Generate profile plot of centroids for k=4
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(basis_km4$centers), max(basis_km4$centers)), xlim = c(0,10), 
     main = "Profile Plot of Basis for Purchase Cluster Centroids (k=4)")
# Label x-axis
axis(1, at = c(1:10), labels = names(basis_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:4))         # k=4
  lines(basis_km4$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,4),"sky blue", "blue"))
# Add cluster names
text(x = 0.5, y = basis_km4$centers[ ,1], 
     labels = paste("Cluster ", c(1:4)), cex = 0.8)

# Generate profile plot of centroids for k=5
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(basis_km5$centers), max(basis_km5$centers)), xlim = c(0,10), 
     main = "Profile Plot of Basis for Purchase Cluster Centroids (k=5)")
# Label x-axis
axis(1, at = c(1:10), labels = names(basis_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:5))         # k=5
  lines(basis_km5$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,3,5),"sky blue", "blue"))
# Add cluster names
text(x = 0.5, y = basis_km5$centers[ ,1], 
     labels = paste("Cluster ", c(1:5)), cex = 0.8)

# k=4 is recommended. Refer to the written report for more explanations.

#------------------------------------------------------------------------------
# Evaluating based on  variables that describe both purchase behavior and  
# basis of purchase
# Combining the variables from the previous two analyses: 
# $ No..of.Brands        : int  3 5 5 2 3 3 4 3 2 4 ...
# $ Brand.Runs           : int  17 25 37 4 6 26 17 8 12 13 ...
# $ Total.Volume         : int  8025 13975 23100 1500 8300 18175 9950 9300 26490
# $ No..of..Trans        : int  24 40 63 4 13 41 26 25 27 18 ...
# $ Value                : num  818 1682 1950 114 591 ...
# $ Avg..Price           : num  10.19 12.03 8.44 7.6 7.12 ...
# $ Others.999           : num  0.492 0.699 0.379 0 0.807 ...
# $ Pur.Vol.No.Promo.... : num  1 0.887 0.942 1 0.614 ...
# $ Pur.Vol.Promo.6..    : num  0 0.0966 0.0195 0 0.1446 ...
# $ Pur.Vol.Other.Promo..: num  0 0.0161 0.039 0 0.241 ...
# $ Pr.Cat.1             : num  0.234 0.293 0.12 0 0 ...
# $ Pr.Cat.2             : num  0.5607 0.5474 0.3182 0.4 0.0482 ...
# $ Pr.Cat.3             : num  0.1308 0.0948 0.5617 0.6 0.1446 ...
# $ Pr.Cat.4             : num  0.0748 0.0644 0 0 0.8072 ...
# $ MaxBrandPur          : num  0.377 0.145 0.545 0.6 0.145 ...
# $ MaxProp              : num  0.502 0.456 0.562 0.6 0.807 ...
# $ MaxPropCat           : num  5 5 14 14 5 5 5 14 5 6 ...

# Copy these variables to a new data frame
both_df <- soap_df[ , c(12:16, 19:22, 31:35, 47:49)]
str(both_df)            # check to ensure all variables have been copied

# Compute normalized distance
both_norm <- sapply(both_df, scale)
row.names(both_norm) <- row.names(both_df)

# Examine the different numbers of clusters (between 2 and 5)
set.seed(123)                 # set seed to make kmeans() reproducible
# Run the k-means algorithm for k=2
both_km2 <- kmeans(both_norm, 2)
# Show size of clusters
both_km2$size
# There is 1 larger cluster and 1 smaller cluster

# Run the k-means algorithm for k=3
both_km3 <- kmeans(both_norm, 3)
# Show size of clusters
both_km3$size
# There are 2 evenly-distributed clusters and a smaller cluster

# Run the k-means algorithm for k=4
both_km4 <- kmeans(both_norm, 4)
# Show size of clusters
both_km4$size
# There are 2 evenly-sized larger clusters and 2 smaller clusters

# Run the k-means algorithm for k=5
both_km5 <- kmeans(both_norm, 5)
# Show size of clusters
both_km5$size
# There are 2 evenly-sized larger clusters, 1 medium-sized cluster and 2 
# smaller clusters

# Check the within-cluster sum of squared distances for each k:
# k = 2
both_km2$withinss
# Compute the within-cluster average sum of squared distances
mean(both_km2$withinss)
# k = 3
both_km3$withinss
# Compute the within-cluster average sum of squared distances
mean(both_km3$withinss)
# k = 4
both_km4$withinss
# Compute the within-cluster average sum of squared distances
mean(both_km4$withinss)
# k = 5
both_km5$withinss
# Compute the within-cluster average sum of squared distances
mean(both_km5$withinss)

# # Use the graphical method to choose the best k:  
# k should be chosen such that the between-cluster distance are highest while 
# the within-cluster sum of squared distances are smallest.

# Plot the average within-cluster sum of squared distances
avg_dist_c <- vector()
for (k in 1:10){
  km_c <- kmeans(both_norm, k)
  avg_dist_c[k] <- mean(km_c$withinss)
}
x <- 1:10
plot(x,avg_dist_c, type = "l",
     xlab = "# of clusters", ylab = "Avg Within-Cluster Distance",
     main = "K vs. Within-Cluster Distance (Both Purchase Behavior & Basis)")

# Moving from 1 to 2 clusters yield significant reductions in within-cluster 
# distance.  Notable improvements can also be seen by changing from 2 to 3, and 
# from 3 to 4 clusters.  From 4 to 5 clusters less improvements can be seen.  
# Adding more clusters beyond 6 brings much less improvement to cluster 
# homogeneity.  Therefore, 5 clusters appear to be the optimal choice according 
# to this plot.     

# Compute the distance between cluster centroids
dist(both_km2$centers)
dist(both_km3$centers)
dist(both_km4$centers)
dist(both_km5$centers)
# Intra-cluster distances are reasonable with no visible outliers

# Generate a plot to characterize the resulting clusters:
n <- ncol(both_df)
# Generate profile plot of centroids for k=2
# First, make an empty scatter plot
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(both_km2$centers), max(both_km2$centers)), xlim = c(0,n), 
     main = "Profile Plot of Purchase Basis & Behavior Cluster Centroids (k=2)")
# Label x-axis
axis(1, at = c(1:n), labels = names(both_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:2))         # k=2
  lines(both_km2$centers[i, ], lty = i, lwd = 2, 
        col = "steel blue")
# Add cluster names
text(x = 0.5, y = both_km2$centers[ ,1], 
     labels = paste("Cluster ", c(1:2)), cex = 0.8)

# Generate profile plot of centroids for k=3
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(both_km3$centers), max(both_km3$centers)), xlim = c(0,n), 
     main = "Profile Plot of Purchase Basis & Behavior Cluster Centroids (k=3)")
# Label x-axis
axis(1, at = c(1:n), labels = names(both_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:3))         # k=3
  lines(both_km3$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,3),"sky blue", "blue"))
# Add legend
legend("topleft", legend = paste("Cluster", c(1:3)), cex = 0.8, lty = 1:3,
       col = c("sky blue","blue","sky blue"))

# Generate profile plot of centroids for k=4
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(both_km4$centers), max(both_km4$centers)), xlim = c(0,n), 
     main = "Profile Plot of Purchase Basis & Behavior Cluster Centroids (k=4)")
# Label x-axis
axis(1, at = c(1:n), labels = names(both_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:4))         # k=4
  lines(both_km4$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,4),"sky blue", "blue"))
# Add legend
legend("topleft", legend = paste("Cluster", c(1:4)), cex = 0.8, lty = 1:4,
       col = c("sky blue","blue","blue","sky blue"))

# Generate profile plot of centroids for k=5
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(both_km5$centers), max(both_km5$centers)), xlim = c(0,n), 
     main = "Profile Plot of Purchase Basis & Behavior Cluster Centroids (k=5)")
# Label x-axis
axis(1, at = c(1:n), labels = names(both_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:5))         # k=5
  lines(both_km5$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,3,5),"sky blue", "blue"))
# Add legend
legend("topleft", legend = paste("Cluster", c(1:5)), cex = 0.8, lty = 1:5,
       col = c("sky blue","blue","sky blue","blue","sky blue"))

# Resulting clusters are less meaningful in this case.  See written report for
# more details.


#------------------------------------------------------------------------------
# Select the most optimal segmentation. Identify the characteristics of these
# clusters. This information can be used to guide the development of 
# advertising and promotional campaigns.

# After comparing cluster sizes, within-cluster average distances, and between-
# cluster distances for the three segmentation methods, conducting market
# segmentation based on purchase behavior (including brand loyalty) using 5 
# clusters appears to be the best strategy.
# Please refer to the written report for more details.

# Add the assigned cluster back to the data frame
soap_df$Cluster <- factor(behavior_km5$cluster)

# Summarize demographics characteristics for each cluster:
# SEC
ggplot(soap_df, aes(x=SEC, fill=Cluster)) + 
  geom_bar(stat="count", width = 0.5, position = position_dodge(0.8)) +
  scale_fill_brewer(palette = "Spectral") +
  ggtitle("Bar Graph of Socio Economic Class by Cluster")
# Age
ggplot(soap_df, aes(x=AGE, fill=Cluster)) + 
  geom_bar(stat="count", width = 0.5, position = position_dodge(0.8)) +
  scale_fill_brewer(palette = "Spectral") +
  ggtitle("Bar Graph of Age by Cluster")
# Education
ggplot(soap_df, aes(x=factor(EDU), fill=Cluster)) + 
  geom_bar(stat="count", width = 0.5, position = position_dodge(0.8)) +
  scale_fill_brewer(palette = "Spectral") +
  ggtitle("Bar Graph of Education by Cluster")
# Children in household
ggplot(soap_df, aes(x=CHILD, fill=Cluster)) + 
  geom_bar(stat="count", width = 0.5, position = position_dodge(0.8)) +
  scale_fill_brewer(palette = "Spectral") +
  ggtitle("Bar Graph of Children in Household by Cluster")

# Household size
summarize_at(group_by(soap_df, Cluster), vars(HS), 
             list(mean = mean, min = min, max = max))
# Average household size & Affluence Index
summarize_at(group_by(soap_df, Cluster), vars(Affluence.Index), 
             list(mean = mean, min = min, max = max))

# Summarize brand loyalty characteristics for each cluster
# Refer to the profile plot of centroids for k=5
par(mar=c(7,3,3,2))
plot(c(0), xaxt = "n", ylab = "", type = "l", 
     ylim = c(min(behavior_km5$centers), max(behavior_km5$centers)), 
     xlim = c(0,11), 
     main = "Profile Plot of Purchase Behavior Cluster Centroids (k=5)")
# Label x-axis
axis(1, at = c(1:11), labels = names(behavior_df), cex.axis = 0.7, las = 2)
# Plot centroids
for (i in c(1:5))         # k=5
  lines(behavior_km5$centers[i, ], lty = i, lwd = 2, 
        col = ifelse(i %in% c(1,3, 5),"sky blue", "blue"))
# Add cluster names
text(x = 0.5, y = behavior_km5$centers[ ,1], 
     labels = paste("Cluster ", c(1:5)), cex = 0.8)

# Summarize basis-for-purchase characteristics for each cluster
summarize_at(group_by(soap_df, Cluster), 
             vars(Pr.Cat.1, Pr.Cat.2,Pr.Cat.3,Pr.Cat.4), 
             mean)
# Selling proposition
ggplot(soap_df, aes(x=factor(MaxPropCat), fill=Cluster)) + 
  geom_bar(stat="count", width = 0.8, position = position_dodge(0.8)) +
  scale_fill_brewer(palette = "Spectral") +
  ggtitle("Bar Graph of Best Selling Proprosition by Cluster")


#------------------------------------------------------------------------------
# Develop a model that classifies the data into these segments 
# First, create a stratified random split of the data set using the 
# createDataPartition() function in the caret package
# First, create an index to split using the 70-30 ratio
set.seed(1234)
train_index <- createDataPartition(soap_df$Cluster, p=0.70, list=FALSE)

# Apply the index to generate training & validation sets
soap_train <- soap_df[train_index,]
soap_valid <- soap_df[-train_index,]

# Fit a classification tree model to the data
soap_tr <- rpart(Cluster ~ ., data = soap_train, 
                 minbucket = 20, method = "class")
soap_tr           # take a look at the tree model
# Plot the decision tree
rpart.plot(soap_tr, type = 1, extra = 1, clip.right.labs = FALSE)

# Prune the tree model
soap_trp <- prune(soap_tr, 
                  cp = soap_tr$cptable[which.min(soap_tr$cptable[ ,"xerror"]),
                                       "CP"])
# Display the pruned tree
rpart.plot(soap_trp, type = 1, extra = 1, clip.right.labs = FALSE)

# Make predictions on validation data
soap_trp_pred <- predict(soap_trp, soap_valid, type = "class")

# Use a confusion matrix to evaluate the accuracy of the tree
confusionMatrix(soap_trp_pred, soap_valid$Cluster)

# For the direct-mail promotion, use a tree model to relate Clustering by 
# Purchase Behavior to Consumer Demographics

# Fit a classification tree model to the data
soap_tr2 <- rpart(Cluster ~ SEC + FEH + MT + SEX + AGE + EDU + HS + CHILD + 
                    Affluence.Index,
                  data = soap_train, 
                  method = "class")

# take a look at the tree model
# Plot the decision tree
rpart.plot(soap_tr2, type = 4, extra = 1, clip.right.labs = FALSE)
soap_tr2

# Prune the tree model
soap_trp2 <- prune(soap_tr2, 
                  cp = soap_tr2$cptable[which.min(soap_tr2$cptable[ ,"xerror"]),
                                       "CP"])
# Display the pruned tree
rpart.plot(soap_trp2, type = 4, extra = 1, clip.right.labs = FALSE)

# Make predictions on validation data
soap_trp2_pred <- predict(soap_trp2, soap_valid, type = "class")

# Use a confusion matrix to evaluate the accuracy of the tree
confusionMatrix(soap_trp2_pred, soap_valid$Cluster)


