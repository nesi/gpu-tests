library(ggplot2)

data <- data.frame(
  name=c("P100","A100-HGX"),
  value=c(980,247.25)
)

# Barplot with reduced gap
ggplot(data, aes(x=name, y=value, fill=name)) + 
  geom_bar(stat = "identity", width = 0.5) +
  scale_x_discrete(labels = c("P100" = "P100", "A100-HGX" = "A100-HGX")) +
  scale_fill_manual(values = c("P100" = "#FF9999", "A100-HGX" = "#66B2FF")) +
  ggtitle("Simulation runtime on Single GPU : A100-HGX vs P100") +  # Add title here
  labs(x="GPU Type", y="Avg. Runtime(s)") +
  theme_minimal() +
  theme(legend.position = "none") +
  coord_cartesian(xlim = c(0.5, 2.5)) # Adjust the x-axis limits

