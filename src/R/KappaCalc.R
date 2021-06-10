install.packages('irr')
library('irr')



preds <- c("Limited", "Safe", "Limited", "Limited", "Safe", "Safe", "Safe", "Safe", "Safe", "Safe", 
           "Limited", "Safe", "Limited", "Safe", "Safe", "Safe", "Limited", "Safe", "Safe", "Safe", "Limited", "Safe", "Safe")

gt <- c('Limited', 
       'Safe', 
       'Limited', 
       'Limited', 
       'Safe', 
       'Limited', 
       'Safe', 
       'Limited', #updated value
       'Limited', 
       'Limited', 
       'Limited', 
       'Safe', 
       'Safe', 
       'Safe', 
       'Limited', 
       'Safe', 
       'Limited', 
       'Limited', 
       'Safe', 
       'Safe', 
       'Limited', 
       'Safe', 
       'Safe')

ratings <- data.frame(gt,preds)



kappa2(ratings) # predefined set of squared weights
#kappa2(anxiety[,1:2], (0:5)^2) # same result with own set of squared weights
# own weights increasing gradually with larger distance from perfect agreement
#kappa2(anxiety[,1:2], c(0,1,2,4,7,11))
data(diagnoses)
# Unweighted Kappa for categorical data without a logical order
kappa2(ratings)

