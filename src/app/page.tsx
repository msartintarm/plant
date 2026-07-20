import EngineCanvas from "@/components/EngineCanvas";
import styles from "./page.module.css";

export default function Home() {
  return (
    <div className={styles.main}>
      <h1 className={styles.title}>Houseplant</h1>
      <EngineCanvas />
    </div>
  );
}
