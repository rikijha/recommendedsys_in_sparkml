package riki.RecommendedSystemDemo;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

import java.util.List;

public class VPPCourseRecommendation {

	public static void main(String[] args) {
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkSession spark = SparkSession.builder()
				.appName("House Price Analysis")
				.config("spark.sql.warehouse.dir","file:///c:/tmp/")
				.master("local[*]").getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("src/main/resources/VPPcourseViews.csv");
		
//		csvData.show();
		
	csvData=csvData.withColumn("proportionWatched", col("proportionWatched").multiply(100));
	
	ALS als=new ALS()
			.setMaxIter(10)
			.setRegParam(0.1).setUserCol("userId").setItemCol("courseId").setRatingCol("proportionWatched");
	
	ALSModel model=als.fit(csvData);
	
	Dataset<Row> userRecs = model.recommendForAllUsers(5);
	List<Row> userRecList=userRecs.takeAsList(5);
	for(Row r:userRecList) {
		int userId=r.getAs(0);
		String recs=r.getAs(1).toString();
		System.out.println("User "+userId+" we might recommend "+recs);
		System.out.println("user has already watched");
		csvData.filter("userId ="+userId).show();
	}
	}
}
