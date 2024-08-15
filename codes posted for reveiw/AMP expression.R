################# R code for plotting four AMP expression across 4 time points upon P. retgerri infection #################

#Load packages
library(ggplot2)
library(gridExtra)

#Id values for four AMPs mentioned in the manuscirpt
#AttA: FBgn0012042
#AttB: FBgn0041581
#DptA: FBgn0004240
#DptB: FBgn0034407

###########################################################################################################################

#Custom-made function for calculation of avergae expression across replicates of the same biological samples 
calculate_averages <- function(vec, n) {
  result <- c()
  for (i in seq(1, length(vec), by = n)) {
    avg <- mean(vec[i:(i + n - 1)])
    result <- c(result, avg)
  }
  return(result)
}

#Input TPM files
TPM = read.csv("..path../TPM.csv", row.names = 1)
#Extract samples for heat-killed P.rettgeri
TPM_Prettgeriheat = TPM[,c(1,2:4,grep("P.rettgeriheat", names(TPM)) )]
#time points at which RNA was extracted
timepoints = c(0, 12, 36, 132)
#Order samples based on time points
TPM_Prettgeriheat = TPM_Prettgeriheat[,c(1,grep("zero", names(TPM_Prettgeriheat)) , grep("twelve", names(TPM_Prettgeriheat)),grep("thirty", names(TPM_Prettgeriheat)),grep("one_three_two", names(TPM_Prettgeriheat)) )]

#Extract TPM for each AMP and calculate avergae expression for each sample across its replicates
#AttacinA
Prettgeriheat_atta = calculate_averages(as.numeric(subset(TPM_Prettgeriheat, TPM_Prettgeriheat$gene_names == "FBgn0012042")[,-1]), n = 3)
#AttacinB
Prettgeriheat_attb = calculate_averages(as.numeric(subset(TPM_Prettgeriheat, TPM_Prettgeriheat$gene_names == "FBgn0041581")[,-1]), n = 3)
#DiptericinA
Prettgeriheat_dpta = calculate_averages(as.numeric(subset(TPM_Prettgeriheat, TPM_Prettgeriheat$gene_names == "FBgn0004240")[,-1]), n = 3)
#DiptericinB
Prettgeriheat_dptb = calculate_averages(as.numeric(subset(TPM_Prettgeriheat, TPM_Prettgeriheat$gene_names == "FBgn0034407")[,-1]), n = 3)

#Data frame of AMPs and time points, which is used for plotting
Prettgeriheat = data.frame(timepoints = timepoints,
                           atta = Prettgeriheat_atta,
                           attb = Prettgeriheat_attb,
                           dpta = Prettgeriheat_dpta,
                           dptb = Prettgeriheat_dptb)

#Use ggplot2 to plot expression of AttacinA
AttacinA_plot = ggplot(Prettgeriheat, aes(x = timepoints, y = atta, group = 1)) +
  geom_line(linetype = "dashed") +       # Make the line dashed
  geom_point() +
  theme_bw() +
  scale_x_continuous(breaks = Prettgeriheat$timepoints) +  # Only show values from timepoints
  labs(x = "Hours post-infection", y = "TPM")+ggtitle("AttacinA")+theme(text = element_text(family = "Times New Roman"))               # Label x and y axes

#Use ggplot2 to plot expression of AttacinB
AttacinB_plot = ggplot(Prettgeriheat, aes(x = timepoints, y = attb, group = 1)) +
  geom_line(linetype = "dashed") +       # Make the line dashed
  geom_point() +
  theme_bw() +
  scale_x_continuous(breaks = Prettgeriheat$timepoints) +  # Only show values from timepoints
  labs(x = "Hours post-infection", y = "TPM")+ggtitle("AttacinB")  +theme(text = element_text(family = "Times New Roman"))             # Label x and y axes

#Use ggplot2 to plot expression of DiptericinA
DiptA_plot = ggplot(Prettgeriheat, aes(x = timepoints, y = dpta, group = 1)) +
  geom_line(linetype = "dashed") +       # Make the line dashed
  geom_point() +
  theme_bw() +
  scale_x_continuous(breaks = Prettgeriheat$timepoints) +  # Only show values from timepoints
  labs(x = "Hours post-infection", y = "TPM")+ggtitle("DiptA")  +theme(text = element_text(family = "Times New Roman")) 


#Use ggplot2 to plot expression of DiptericinB
DiptB_plot = ggplot(Prettgeriheat, aes(x = timepoints, y = dptb, group = 1)) +
  geom_line(linetype = "dashed") +       # Make the line dashed
  geom_point() +
  theme_bw() +
  scale_x_continuous(breaks = Prettgeriheat$timepoints) +  # Only show values from timepoints
  labs(x = "Hours post-infection", y = "TPM")+ggtitle("DiptB")  +theme(text = element_text(family = "Times New Roman")) 

#Combine and plot graphs in a single panel
grid.arrange(AttacinA_plot, AttacinB_plot, DiptA_plot, DiptB_plot, ncol = 2) 
