import pandas as pd


weather_df = pd.read_csv('./data/1314_with_weather.csv')
site_df = pd.read_csv('./data/1314_site_table_final.csv')
weather_cols = [
    "SITE_ID",
    "DailyAverageDryBulbTemperature",
    "DailyMaximumDryBulbTemperature",
    "DailyMinimumDryBulbTemperature",
    "DailyPrecipitation",
    "DailyAverageRelativeHumidity",
    'Nitrate', 
    'Nitrogen', 
    'Phosphorus', 
    'Specific conductance', 
    'Temperature, water', 
    'Turbidity', 
    'pH'
    
]

weather_df = weather_df[weather_cols]
cols = [
    'Total_Individuals',
    'Taxa_Richness',
    'Shannon_Index',
    'Simpson_Index',
    'EPT_Individuals',
    'EPT_Taxa',
    'EPT_%',
    'LAT_DD83',
    'LON_DD83',
    'YEAR',
    'Sampling_Month',
    'Nitrate',
    'Nitrogen',
    'Phosphorus',
    'Specific conductance',
    'Temperature, water',
    'Turbidity',
    'pH',
    'DailyAverageDryBulbTemperature',
    'DailyMaximumDryBulbTemperature',
    'DailyMinimumDryBulbTemperature',
    'DailyPrecipitation',
    'DailyAverageRelativeHumidity',
    'Taxa_3406','Taxa_3407','Taxa_3408','Taxa_3418','Taxa_3439',
    'Taxa_3450','Taxa_3458','Taxa_3459','Taxa_3465','Taxa_3472',
    'Taxa_3476','Taxa_3481','Taxa_3487','Taxa_3490','Taxa_3491',
    'Taxa_3496','Taxa_3500','Taxa_3501','Taxa_3511','Taxa_3526',
    'Taxa_3532','Taxa_3535','Taxa_3540','Taxa_3542','Taxa_3544',
    'Taxa_3546','Taxa_3548','Taxa_3556','Taxa_3564','Taxa_3565',
    'Taxa_3566','Taxa_3567','Taxa_3578','Taxa_3579','Taxa_3581',
    'Taxa_3582','Taxa_3583','Taxa_3587','Taxa_3590','Taxa_3593',
    'Taxa_3594','Taxa_3600','Taxa_3606','Taxa_3612','Taxa_3614',
    'Taxa_3620','Taxa_3622','Taxa_3623','Taxa_3625','Taxa_3628',
    'Taxa_3641','Taxa_3646','Taxa_3657','Taxa_3658','Taxa_3661',
    'Taxa_3665','Taxa_3676','Taxa_3678','Taxa_3679','Taxa_3688',
    'Taxa_3695','Taxa_3697','Taxa_3698','Taxa_3703','Taxa_3704',
    'Taxa_3705','Taxa_3706','Taxa_3711','Taxa_3717','Taxa_3722',
    'Taxa_3726','Taxa_3729','Taxa_3735','Taxa_3744','Taxa_3745',
    'Taxa_3752','Taxa_3753','Taxa_3756','Taxa_3759','Taxa_3771',
    'Taxa_3783','Taxa_3789','Taxa_3792','Taxa_3799','Taxa_3802',
    'Taxa_3809','Taxa_3811','Taxa_3814','Taxa_3816','Taxa_3817',
    'Taxa_3821','Taxa_3832','Taxa_3844','Taxa_3845','Taxa_3846',
    'Taxa_3847','Taxa_3855','Taxa_3866','Taxa_3882','Taxa_3890',
    'Taxa_3892','Taxa_3895','Taxa_3897','Taxa_3904','Taxa_3911',
    'Taxa_3912','Taxa_3918','Taxa_3920','Taxa_3925','Taxa_3928',
    'Taxa_3939','Taxa_3941','Taxa_3945','Taxa_3948','Taxa_3952',
    'Taxa_3967','Taxa_3977','Taxa_3980','Taxa_3981','Taxa_3982',
    'Taxa_3986','Taxa_3997','Taxa_4000','Taxa_4004','Taxa_4005',
    'Taxa_4011','Taxa_4028','Taxa_4038','Taxa_4041','Taxa_4042',
    'Taxa_4043','Taxa_4056','Taxa_4060','Taxa_4071','Taxa_4082',
    'Taxa_4085','Taxa_4088','Taxa_4099','Taxa_4101','Taxa_4102',
    'Taxa_4106','Taxa_4107','Taxa_4108','Taxa_4112','Taxa_4120',
    'Taxa_4121','Taxa_4130','Taxa_4134','Taxa_4138','Taxa_4141',
    'Taxa_4143','Taxa_4149','Taxa_4158','Taxa_4159','Taxa_4162',
    'Taxa_4166','Taxa_4170','Taxa_4174','Taxa_4175','Taxa_4178',
    'Taxa_4187','Taxa_4188','Taxa_4191','Taxa_4192','Taxa_4199',
    'Taxa_4203','Taxa_4206','Taxa_4209','Taxa_4212','Taxa_4213',
    'Taxa_4223','Taxa_4245','Taxa_4247','Taxa_4249','Taxa_4253',
    'Taxa_4260','Taxa_4263','Taxa_4276','Taxa_4280','Taxa_4281',
    'Taxa_4288','Taxa_4296','Taxa_4298','Taxa_4300','Taxa_4307',
    'Taxa_4308','Taxa_4309','Taxa_4310','Taxa_4311','Taxa_4315',
    'Taxa_4327','Taxa_4332','Taxa_4337','Taxa_4340','Taxa_4347',
    'Taxa_4348','Taxa_4361','Taxa_4362','Taxa_4368','Taxa_4369',
    'Taxa_4371','Taxa_4379','Taxa_4382','Taxa_4384','Taxa_4391',
    'Taxa_4393','Taxa_4394','Taxa_4400','Taxa_4412','Taxa_4428',
    'Taxa_4430','Taxa_4433',
    'STRAH_CAT_LargeStreams',
    'STRAH_CAT_Missing',
    'STRAH_CAT_RiversMajor',
    'STRAH_CAT_RiversOther',
    'STRAH_CAT_SmallStreams',
    'EPA_REG_Missing','EPA_REG_Region_1','EPA_REG_Region_10',
    'EPA_REG_Region_2','EPA_REG_Region_3','EPA_REG_Region_4',
    'EPA_REG_Region_5','EPA_REG_Region_6','EPA_REG_Region_7',
    'EPA_REG_Region_8','EPA_REG_Region_9',
    'Season_Fall',
    'Season_Spring',
    'Season_Summer',
    'Season_Unknown',
    'MMI_BENT_MinMax',
    'BENT_MMI_COND_enc'
]

one_hot_cols = [
    'STRAH_CAT_LargeStreams',
    'STRAH_CAT_Missing',
    'STRAH_CAT_RiversMajor',
    'STRAH_CAT_RiversOther',
    'STRAH_CAT_SmallStreams',
    'EPA_REG_Missing','EPA_REG_Region_1','EPA_REG_Region_10',
    'EPA_REG_Region_2','EPA_REG_Region_3','EPA_REG_Region_4',
    'EPA_REG_Region_5','EPA_REG_Region_6','EPA_REG_Region_7',
    'EPA_REG_Region_8','EPA_REG_Region_9',
    'Season_Fall',
    'Season_Spring',
    'Season_Summer',
    'Season_Unknown',
]
weather_agg = (
    weather_df
    .groupby("SITE_ID", as_index=False)
    .mean()
)
merged_df = pd.merge(site_df, weather_agg, on='SITE_ID', how='left')
merged_df = merged_df[cols]
merged_df.to_csv('./data/all_merged_site_data.csv', index=False)

print(merged_df.shape)

