#pragma once

/**
 * @brief The Observer interface for four notification types.
 *
 * @tparam T1
 * @tparam void
 * @tparam void
 * @tparam void
 */
template <typename T1, typename T2 = void, typename T3 = void,
          typename T4 = void>
class Observer {
 public:
  virtual ~Observer() {}
  virtual void notify(T1) = 0;
  virtual void notify(T2) = 0;
  virtual void notify(T3) = 0;
  virtual void notify(T4) = 0;
};

/**
 * @brief The Observer interface for three notification types.
 *
 * @tparam T1
 * @tparam T2
 * @tparam T3
 */
template <typename T1, typename T2, typename T3>
class Observer<T1, T2, T3> {
 public:
  virtual ~Observer() {}
  virtual void notify(T1) = 0;
  virtual void notify(T2) = 0;
  virtual void notify(T3) = 0;
};

/**
 * @brief The Observer interface for two notification types.
 *
 * @tparam T1
 * @tparam T2
 */
template <typename T1, typename T2>
class Observer<T1, T2> {
 public:
  virtual ~Observer() {}
  virtual void notify(T1) = 0;
  virtual void notify(T2) = 0;
};

/**
 * @brief The Observer interface for one notification type.
 *
 * @tparam T1
 */
template <typename T1>
class Observer<T1> {
 public:
  virtual ~Observer() {}
  virtual void notify(T1) = 0;
};
