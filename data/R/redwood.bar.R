df_p = data.frame('value' = c(4.88, 5.63, 4.65, 5.02), 'type'= rep(c('Case 1', 'Case 2'), 2), 'data'=c(rep('in-sample', 2), rep('out-sample', 2)))
library(ggplot2)
ggplot(df_p, aes(fill=data, y=value, x=type)) + 
  geom_bar(position="dodge", stat="identity") + coord_cartesian(ylim=c(4.1,5.9)) + labs(x = '', y = 'average mean intensity') +
  geom_text(aes(label=value), position=position_dodge(width=0.9), vjust=-0.25)

