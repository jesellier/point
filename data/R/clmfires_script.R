library(spatstat)

# rescale ppp 
clmfires2 = rescale(clmfires, 200)
clmfires2 = shift(clmfires2, c(-1, -1))

# axes=TRUE: add axix
# legend=FALSE: remove legend
# main = '': remove title
# pch = rep(19, 4): specifies same pattern for different fire causes
plot(clmfires2, which.marks="cause", pch = rep(19, 4), axes=TRUE, cex=0.25, legend=FALSE, main = '')


# create new ppp by subsetting current one
clmfires3 = clmfires2
clmfires3 = subset(clmfires3, cause == 'lightning' & date > "2004-01-01")
#clmfires3 = subset(clmfires3, date > "2004-01-01")


plot(clmfires3, which.marks="cause", pch = rep(19, 4), axes=TRUE, cex=0.25, cols = "red", legend=FALSE, main = '')
