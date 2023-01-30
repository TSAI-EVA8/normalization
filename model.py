import torch.nn as nn
import torch.nn.functional as F
import config as cf

class Net(nn.Module):
    def __init__(self,use_norm):
        super(Net, self).__init__()
        self.use_norm=use_norm
        # Input Block
        
        if self.use_norm=='BATCH':
          norm1=nn.BatchNorm2d(14) ## adding self.norm1 adds the paramtere two times
        elif self.use_norm=='LAYER':
          norm1=nn.LayerNorm([14,26,26])
        else:# self.use_norm=='GROUP':
          norm1=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            norm1,
            nn.Dropout(cf.DROPOUT_VALUE)
        ) # output_size = 26

        
        # CONVOLUTION BLOCK 1
        
        if self.use_norm=='BATCH':
          norm2=nn.BatchNorm2d(28)
        elif self.use_norm=='LAYER':
          norm2=nn.LayerNorm([28,24,24])
        elif self.use_norm=='GROUP':
          norm2=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,28)


        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=28, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            norm2,
            nn.Dropout(cf.DROPOUT_VALUE)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        if self.use_norm=='BATCH':
          norm3=nn.BatchNorm2d(14)
          norm4=nn.BatchNorm2d(12)
          norm5=nn.BatchNorm2d(12)
          norm6=nn.BatchNorm2d(12)
        
        elif self.use_norm=='LAYER':
          norm3=nn.LayerNorm([14,10,10])
          norm4=nn.LayerNorm([12,8,8])
          norm5=nn.LayerNorm([12,6,6])
          norm6=nn.LayerNorm([12,6,6])
          
        elif self.use_norm=='GROUP':
          norm3=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,14)
          norm4=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,12)
          norm5=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,12)
          norm6=nn.GroupNorm (cf.NUM_GROUPS_FOR_GN,12)
        
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            norm3,
            nn.Dropout(cf.DROPOUT_VALUE)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            norm4,
            nn.Dropout(cf.DROPOUT_VALUE)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            norm5,
            nn.Dropout(cf.DROPOUT_VALUE)
        ) # output_size = 6

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            norm6,
            nn.Dropout(cf.DROPOUT_VALUE)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

        self.dropout = nn.Dropout(cf.DROPOUT_VALUE)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
