#pragma once

#include <QAction>
#include <QApplication>
#include <QByteArray>
#include <QCoreApplication>
#include <QDesktopWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QIcon>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QObject>
#include <QPlainTextEdit>
#include <QRect>
#include <QSessionManager>
#include <QSettings>
#include <QSettings>
#include <QStatusBar>
#include <QString>
#include <QTextStream>
#include <QToolBar>

QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
class QPlainTextEdit;
class QSessionManager;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow();

  void loadFile(const QString &fileName);

protected:
  void closeEvent(QCloseEvent *event) override;

private slots:
  void newFile();
  void open();
  bool save();
  bool saveAs();
  void about();
  void documentWasModified();
#ifndef QT_NO_SESSIONMANAGER
  void commitData(QSessionManager &);
#endif

private:
  void createActions();
  void createStatusBar();
  void readSettings();
  void writeSettings();
  bool maybeSave();
  bool saveFile(const QString &fileName);
  void setCurrentFile(const QString &fileName);
  QString strippedName(const QString &fullFileName);

  QPlainTextEdit *textEdit;
  QString curFile;
};
